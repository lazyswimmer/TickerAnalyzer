"""
Enhanced Stock Assessment Script
--------------------------------
This expands the original Yahoo-Finance-based prototype into a broader
stock-intelligence framework.

It still works without paid API keys, but some modules are best-effort:
- SEC filings use the public SEC submissions API.
- Macro data uses pandas_datareader/FRED if installed and available.
- Analyst/options/insider data uses yfinance where available.
- Alternative data and deep transcript analysis are provided as structured
  placeholders unless you connect a specialized data provider.

Install common dependencies:
    pip install yfinance pandas numpy financedatabase requests pandas_datareader

Run:
    python enhanced_stock_assessment.py
"""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from financedatabase import Equities

try:
    from pandas_datareader import data as pdr
except Exception:
    pdr = None


# ============================================================
# Helper utilities
# ============================================================

SEC_USER_AGENT = "stock-research-script/1.0 your-email@example.com"


def safe_call(func, default=None, label: str = ""):
    """Run a data call without crashing the whole assessment if one source fails."""
    try:
        return func()
    except Exception as exc:
        return {
            "error": str(exc),
            "source_failed": label or getattr(func, "__name__", "unknown"),
        }


# ------------------------------------------------------------
# Shared yfinance Ticker cache
# ------------------------------------------------------------
# Every analysis function below used to call ``yf.Ticker(ticker)`` and hit the
# network again (``.info`` alone fired ~16 separate cold requests per run). That
# made an assessment take ~90s and, worse, made the FIRST request for a ticker
# fail intermittently: yfinance needs a Yahoo crumb/cookie handshake on its first
# call, which is only cached *after* that first attempt, so the first load would
# error and the second would succeed.
#
# We now share a single Ticker per assessment (yfinance caches .info/.history/etc.
# on the object) and prime it with a small retry so the cold handshake self-heals.

_TICKER_CACHE: Dict[str, yf.Ticker] = {}


def get_ticker(ticker: str) -> yf.Ticker:
    """Return a process-cached yf.Ticker so all analysis functions share one
    network-backed object instead of each re-fetching .info over the network."""
    ticker = ticker.upper().strip()
    cached = _TICKER_CACHE.get(ticker)
    if cached is None:
        cached = yf.Ticker(ticker)
        _TICKER_CACHE[ticker] = cached
    return cached


# ------------------------------------------------------------
# Cached, throttled access to Yahoo's quoteSummary (.info) endpoint
# ------------------------------------------------------------
# .info is the most aggressively rate-limited Yahoo endpoint, and a full
# assessment used to fire ~20 uncached calls at it in a tight burst (the
# target plus every peer snapshot). From a cloud server's shared IP that
# eventually trips Yahoo's throttle — at which point every .info-backed field
# (name, market cap, P/E, margins, the whole peer comparison) silently
# vanishes while statements/prices/FRED keep working. Defenses, in order:
#   1. cache results per process for an hour — peer sets overlap heavily
#      across analyses, so repeat traffic drops ~90%
#   2. space real fetches out and retry with backoff instead of bursting
#   3. circuit-break after repeated failures so a blocked period degrades
#      fast instead of stacking retry sleeps
_INFO_CACHE: Dict[str, Any] = {}
_INFO_CACHE_TTL = 60 * 60
_INFO_PACING = {"last_fetch": 0.0}
_INFO_MIN_INTERVAL = 0.35
_INFO_BREAKER = {"consecutive_failures": 0, "cooldown_until": 0.0}


def get_cached_info(ticker: str) -> Dict[str, Any]:
    """The one sanctioned way to read .info. Returns {} when Yahoo won't
    cooperate — callers already treat missing fields as absent data."""
    ticker = ticker.upper().strip()
    now = time.time()

    hit = _INFO_CACHE.get(ticker)
    if hit is not None and now - hit[0] < _INFO_CACHE_TTL:
        return hit[1]

    if now < _INFO_BREAKER["cooldown_until"]:
        return {}

    for attempt in range(3):
        pause = _INFO_PACING["last_fetch"] + _INFO_MIN_INTERVAL - time.time()
        if pause > 0:
            time.sleep(pause)
        _INFO_PACING["last_fetch"] = time.time()
        try:
            info = get_ticker(ticker).info
            # A throttled fetch often returns a near-empty dict rather than
            # raising; only accept a substantive payload.
            if isinstance(info, dict) and len(info) > 5:
                _INFO_CACHE[ticker] = (time.time(), info)
                _INFO_BREAKER["consecutive_failures"] = 0
                return info
        except Exception:
            pass
        # Drop the possibly-poisoned Ticker so a retry redoes the handshake.
        _TICKER_CACHE.pop(ticker, None)
        if attempt < 2:
            time.sleep(1.0 * (attempt + 1))

    _INFO_BREAKER["consecutive_failures"] += 1
    if _INFO_BREAKER["consecutive_failures"] >= 3:
        _INFO_BREAKER["cooldown_until"] = time.time() + 120
        print("[yahoo-info] quoteSummary appears throttled; failing fast for 2 minutes")
    return {}


def prime_ticker(ticker: str, attempts: int = 3, delay: float = 1.0) -> None:
    """Warm up the yfinance session for ``ticker`` before the real work starts.
    Never raises: if priming fails, per-source safe_call handling still
    produces a (partial) result. Delegates to get_cached_info, which owns
    retry/backoff/caching for the .info endpoint."""
    get_cached_info(ticker)


def clean_number(value: Any) -> Optional[float]:
    """Convert numeric-looking values to float, otherwise return None."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    try:
        return float(value)
    except Exception:
        return None


def latest_value(df: pd.DataFrame, row_name: str) -> Optional[float]:
    """Get the most recent value from a yfinance financial statement row."""
    try:
        if df is None or df.empty or row_name not in df.index:
            return None
        series = df.loc[row_name].dropna()
        if series.empty:
            return None
        return clean_number(series.iloc[0])
    except Exception:
        return None


def ttm_value(
    quarterly_df: pd.DataFrame,
    annual_df: pd.DataFrame,
    row_name: str,
) -> Optional[float]:
    """Trailing-twelve-month value of a *flow* row (revenue, FCF, capex, ...):
    the sum of the four most recent quarterly columns.

    Falls back to the latest annual figure when fewer than four quarters are
    available (some foreign and small-cap tickers report semi-annually, and
    newly listed ones have short histories). The annual number can be up to
    ~14 months stale, but it beats returning nothing.
    """
    try:
        if (
            quarterly_df is not None
            and not quarterly_df.empty
            and row_name in quarterly_df.index
        ):
            series = quarterly_df.loc[row_name].dropna()
            if len(series) >= 4:
                # yfinance orders columns most-recent-first.
                return clean_number(series.iloc[:4].sum())
    except Exception:
        pass
    return latest_value(annual_df, row_name)


def mrq_value(
    quarterly_df: pd.DataFrame,
    annual_df: pd.DataFrame,
    row_name: str,
) -> Optional[float]:
    """Most-recent-quarter value of a *stock* row (receivables, inventory, ...).
    Point-in-time balance-sheet items should never be summed across quarters —
    the freshest single column is the right reading. Falls back to annual."""
    value = latest_value(quarterly_df, row_name)
    return value if value is not None else latest_value(annual_df, row_name)


def flatten_yfinance_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten MultiIndex columns that yfinance sometimes returns."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            "_".join([str(x) for x in col if str(x) != ""]).strip("_")
            for col in df.columns
        ]
    return df

def make_json_safe(value: Any) -> Any:
    """
    Recursively convert pandas/numpy/yfinance objects into JSON-safe Python types.

    This allows the final dictionary to be returned by an API or saved as JSON
    for an HTML page to consume.
    """
    if value is None:
        return None

    # pandas DataFrame
    if isinstance(value, pd.DataFrame):
        if value.empty:
            return []
        df = value.copy()
        df = df.reset_index()

        # Convert timestamps and other objects safely
        for col in df.columns:
            df[col] = df[col].map(make_json_safe)

        return df.to_dict(orient="records")

    # pandas Series
    if isinstance(value, pd.Series):
        if value.empty:
            return {}
        return {
            make_json_safe(k): make_json_safe(v)
            for k, v in value.to_dict().items()
        }

    # pandas Timestamp / datetime-like
    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    # numpy scalar values
    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        if np.isnan(value) or np.isinf(value):
            return None
        return float(value)

    if isinstance(value, np.bool_):
        return bool(value)

    # dict
    if isinstance(value, dict):
        return {
            str(make_json_safe(k)): make_json_safe(v)
            for k, v in value.items()
        }

    # list / tuple / set
    if isinstance(value, (list, tuple, set)):
        return [make_json_safe(v) for v in value]

    # yfinance option-chain object or other namedtuple-like objects
    if hasattr(value, "_asdict"):
        return make_json_safe(value._asdict())

    # objects with date/datetime isoformat
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:
            pass

    # normal JSON-safe primitives
    if isinstance(value, (str, int, float, bool)):
        return value

    # final fallback
    return str(value)

# ============================================================
# 1. Quantitative data: price, financials, ratios, valuation
# ============================================================

def retrieve_quantitative_data(ticker: str) -> Dict[str, Any]:
    stock = get_ticker(ticker)

    return {
        "price_history": stock.history(period="5y"),
        "info": get_cached_info(ticker),
        "balance_sheet": stock.balance_sheet,
        "income_statement": stock.income_stmt,
        "cashflow": stock.cashflow,
        "quarterly_financials": stock.quarterly_financials,
        "quarterly_balance_sheet": stock.quarterly_balance_sheet,
        "quarterly_cashflow": stock.quarterly_cashflow,
        "major_holders": stock.major_holders,
        "institutional_holders": stock.institutional_holders,
        "mutualfund_holders": getattr(stock, "mutualfund_holders", None),
        "sustainability": getattr(stock, "sustainability", None),
    }


# ============================================================
# 2. News and basic sentiment inputs
# ============================================================

def retrieve_news_data(ticker: str) -> pd.DataFrame:
    stock = get_ticker(ticker)
    news = stock.news or []
    return pd.DataFrame(news)


def retrieve_basic_news_sentiment(ticker: str) -> Dict[str, Any]:
    """
    Lightweight sentiment proxy based on headline keywords.
    This is not a real NLP model; it is a simple first-pass signal.
    """
    news_df = retrieve_news_data(ticker)

    positive_words = {
        "beat", "beats", "growth", "upgrade", "raised", "surge", "record",
        "profit", "strong", "bullish", "outperform", "expands", "launches",
    }

    negative_words = {
        "miss", "misses", "downgrade", "cut", "falls", "lawsuit", "probe",
        "weak", "bearish", "underperform", "decline", "layoffs", "warning",
    }

    scored = []

    if not news_df.empty:
        title_col = "title" if "title" in news_df.columns else None

        if title_col:
            for _, row in news_df.iterrows():
                title = str(row.get(title_col, ""))
                tokens = set(title.lower().replace("-", " ").split())

                pos = len(tokens & positive_words)
                neg = len(tokens & negative_words)

                scored.append({
                    "title": title,
                    "positive_hits": pos,
                    "negative_hits": neg,
                    "simple_sentiment_score": pos - neg,
                })

    sentiment_df = pd.DataFrame(scored)

    return {
        "raw_news": news_df,
        "headline_keyword_sentiment": sentiment_df,
        "average_simple_score": (
            sentiment_df["simple_sentiment_score"].mean()
            if not sentiment_df.empty
            else None
        ),
    }


# ============================================================
# 3. Technical indicators: RSI, MACD, moving averages, volatility
# ============================================================

def retrieve_technical_indicators(ticker: str) -> pd.DataFrame:
    df = yf.download(ticker, period="5y", auto_adjust=False, progress=False)
    df = flatten_yfinance_columns(df)

    close_candidates = [
        c for c in df.columns
        if c == "Close" or c.startswith("Close_")
    ]

    if not close_candidates:
        return df

    close_col = close_candidates[0]

    df["SMA_20"] = df[close_col].rolling(20).mean()
    df["SMA_50"] = df[close_col].rolling(50).mean()
    df["SMA_200"] = df[close_col].rolling(200).mean()

    delta = df[close_col].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))

    df["EMA12"] = df[close_col].ewm(span=12, adjust=False).mean()
    df["EMA26"] = df[close_col].ewm(span=26, adjust=False).mean()
    df["MACD"] = df["EMA12"] - df["EMA26"]
    df["Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    returns = df[close_col].pct_change()

    df["Volatility_20d_annualized"] = returns.rolling(20).std() * np.sqrt(252)
    df["Volatility_60d_annualized"] = returns.rolling(60).std() * np.sqrt(252)

    df["Cumulative_Return"] = (1 + returns.fillna(0)).cumprod() - 1
    df["Rolling_Max"] = df[close_col].cummax()
    df["Drawdown"] = df[close_col] / df["Rolling_Max"] - 1

    return df


# ============================================================
# 4. Broad market, sector, rates, and macro data
# ============================================================

def retrieve_broad_market_data() -> pd.DataFrame:
    tickers = [
        "SPY", "QQQ", "IWM", "DIA",
        "XLF", "XLK", "XLY", "XLE", "XLI",
        "XLV", "XLP", "XLU", "XLRE", "XLB",
        "TLT", "IEF", "HYG", "LQD", "UUP", "GLD", "USO",
    ]

    data = yf.download(tickers, period="5y", auto_adjust=False, progress=False)

    if isinstance(data.columns, pd.MultiIndex) and "Close" in data.columns.get_level_values(0):
        return data["Close"]

    return data


def retrieve_macro_data(start: str = "2015-01-01") -> Dict[str, Any]:
    """
    Pull major macro series from FRED if pandas_datareader is installed.
    If not installed, returns instructions instead of failing.
    """
    fred_series = {
        "fed_funds_rate": "FEDFUNDS",
        "ten_year_treasury": "DGS10",
        "two_year_treasury": "DGS2",
        "cpi": "CPIAUCSL",
        "unemployment_rate": "UNRATE",
        "real_gdp": "GDPC1",
        "investment_grade_spread": "BAMLC0A0CM",
        "high_yield_spread": "BAMLH0A0HYM2",
        "consumer_sentiment": "UMCSENT",
    }

    if pdr is None:
        return {
            "error": "pandas_datareader is not installed. Run: pip install pandas_datareader",
            "requested_series": fred_series,
        }

    # These 9 series are the single biggest cost in the whole assessment (~7-8s
    # each, ~68s sequentially). They are independent and hit FRED — a separate,
    # robust backend, not Yahoo's crumb-protected session — so fetching them
    # concurrently is safe and collapses ~68s down to roughly one series' latency.
    output: Dict[str, Any] = {}

    def fetch(code: str):
        return pdr.DataReader(code, "fred", start)

    with ThreadPoolExecutor(max_workers=len(fred_series)) as pool:
        futures = {
            name: pool.submit(safe_call, lambda code=code: fetch(code), label=f"FRED:{code}")
            for name, code in fred_series.items()
        }
        for name, future in futures.items():
            output[name] = future.result()  # safe_call never raises

    return output


# ------------------------------------------------------------
# Macro regime (a cheap, cached signal the per-stock overlay reacts to)
# ------------------------------------------------------------
# Macro state is identical for every ticker and changes slowly, so we fetch a
# small, fast set of indicators ONCE and cache them, rather than paying for macro
# on every request. This is what makes a macro overlay viable on the web path.

_MACRO_REGIME_CACHE: Dict[str, Any] = {"data": None, "fetched_at": 0.0}
_MACRO_REGIME_TTL_SECONDS = 6 * 60 * 60  # refresh a few times a day


def _fred_latest(code: str, start: str = "2018-01-01") -> Optional[float]:
    """Return the most recent value of a single FRED series, or None."""
    if pdr is None:
        return None
    df = pdr.DataReader(code, "fred", start)
    series = df[code].dropna() if code in df.columns else df.iloc[:, 0].dropna()
    return clean_number(series.iloc[-1]) if not series.empty else None


def get_macro_regime() -> Dict[str, Any]:
    """
    Distill a few macro indicators into a 'regime' the per-stock risk overlay can
    react to. Cached for several hours (same for every ticker, changes slowly).

    Uses only fast series and takes the 10y-2y curve from the published T10Y2Y
    spread, deliberately avoiding the daily DGS10/DGS2 series, which read-time-out.
    """
    now = time.time()
    if (
        _MACRO_REGIME_CACHE["data"] is not None
        and (now - _MACRO_REGIME_CACHE["fetched_at"]) < _MACRO_REGIME_TTL_SECONDS
    ):
        return _MACRO_REGIME_CACHE["data"]

    series_codes = {
        "fed_funds": "FEDFUNDS",
        "yield_curve_10y_2y": "T10Y2Y",
        "high_yield_spread": "BAMLH0A0HYM2",
    }

    # Fetch concurrently with a hard per-series timeout. pandas_datareader has no
    # timeout knob and some series (notably treasuries) can hang ~30s; here a stuck
    # series is abandoned after a few seconds and simply omitted from the regime.
    # shutdown(wait=False) means a lingering fetch never blocks the response; the
    # result is cached for hours, so this cost is paid at most a few times a day.
    values: Dict[str, Optional[float]] = {}
    pool = ThreadPoolExecutor(max_workers=len(series_codes))
    try:
        futures = {
            name: pool.submit(_fred_latest, code)
            for name, code in series_codes.items()
        }
        for name, future in futures.items():
            try:
                res = future.result(timeout=10)
                values[name] = res if isinstance(res, (int, float)) else None
            except Exception:
                values[name] = None
    finally:
        pool.shutdown(wait=False)

    fed_funds = values.get("fed_funds")
    curve = values.get("yield_curve_10y_2y")
    hy_spread = values.get("high_yield_spread")

    flags = {
        "curve_inverted": curve is not None and curve < 0,
        "rates_elevated": fed_funds is not None and fed_funds >= 4.0,
        "credit_stress": hy_spread is not None and hy_spread >= 5.0,
    }

    parts = []
    if fed_funds is not None:
        parts.append(f"Fed funds {fed_funds:.2f}%")
    if curve is not None:
        parts.append(f"10y-2y {curve:+.2f}%{' (inverted)' if curve < 0 else ''}")
    if hy_spread is not None:
        parts.append(f"HY credit spread {hy_spread:.2f}%")
    label = "; ".join(parts) if parts else "Macro data unavailable"

    regime = {
        "fed_funds_rate": fed_funds,
        "yield_curve_10y_2y": curve,
        "high_yield_spread": hy_spread,
        "flags": flags,
        "label": label,
    }

    _MACRO_REGIME_CACHE["data"] = regime
    _MACRO_REGIME_CACHE["fetched_at"] = now
    return regime


# ============================================================
# 5. Industry and competitor data
# ============================================================

_PEER_UNIVERSE_CACHE: Dict[str, Any] = {"df": None}
_PEER_UNIVERSE_CSV = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "peer_universe.csv"
)


def _load_peer_universe() -> pd.DataFrame:
    """US-equity classification table for peer discovery (symbol → name,
    sector, industry, market-cap bucket).

    Prefers the bundled peer_universe.csv snapshot. financedatabase's live
    dataset is a large runtime download into pandas that quietly fails on
    small servers (memory/network) — which is exactly why the deployed site
    showed no peer comparison while local runs worked. The snapshot is
    regenerated by re-running the export (see repo history) whenever it feels
    stale; sector/industry classifications change rarely. Falls back to the
    live dataset only if the file is missing. Cached per process.
    """
    if _PEER_UNIVERSE_CACHE["df"] is not None:
        return _PEER_UNIVERSE_CACHE["df"]

    df = None
    try:
        if os.path.exists(_PEER_UNIVERSE_CSV):
            df = pd.read_csv(_PEER_UNIVERSE_CSV, index_col="symbol")
    except Exception as exc:
        print(f"[peer_universe] bundled snapshot failed to load: {exc}")

    if df is None or df.empty:
        try:
            df = Equities().select()
        except Exception as exc:
            print(f"[peer_universe] financedatabase fallback failed: {exc}")
            df = pd.DataFrame()

    _PEER_UNIVERSE_CACHE["df"] = df
    return df


def _loose_taxonomy_match(series: pd.Series, label: Optional[str]) -> Optional[str]:
    """Find the closest value in a financedatabase classification column for a
    label from yfinance's different taxonomy (e.g. yfinance 'Technology' →
    financedatabase 'Information Technology'). Tries exact (case-insensitive),
    then substring containment, then shared-word overlap. Returns a value from
    the series' own vocabulary, or None."""
    if not label:
        return None
    target = str(label).lower().strip()
    try:
        values = [str(v) for v in series.dropna().unique()]
    except Exception:
        return None

    for v in values:
        if v.lower() == target:
            return v
    for v in values:
        vl = v.lower()
        if target in vl or vl in target:
            return v

    def tokens(text: str) -> set:
        return {t for t in text.replace("&", " ").replace(",", " ").replace("-", " ").split() if len(t) > 4}

    target_tokens = tokens(target)
    best, best_overlap = None, 0
    for v in values:
        overlap = len(target_tokens & tokens(v.lower()))
        if overlap > best_overlap:
            best, best_overlap = v, overlap
    return best


def retrieve_industry_competitor_data(
    ticker: str,
    max_competitors: int = 30
) -> Dict[str, Any]:
    ticker = ticker.upper().strip()

    df = _load_peer_universe()

    if not isinstance(df, pd.DataFrame) or df.empty:
        return {"sector": None, "industry": None, "competitors": pd.DataFrame()}

    # IMPORTANT: yfinance and financedatabase use different sector/industry
    # taxonomies (yfinance: "Technology"/"Consumer Electronics"; financedatabase:
    # "Information Technology"/"Electronic Equipment, ..."). Filtering financedatabase
    # by yfinance's labels matches nothing. So look the ticker up *inside*
    # financedatabase and filter by ITS own classification, keeping everything in
    # one taxonomy. Fall back to yfinance's labels only if the ticker isn't listed.
    fdb_sector = fdb_industry = fdb_cap = None
    if ticker in df.index:
        row = df.loc[ticker]
        if isinstance(row, pd.DataFrame):  # duplicate index → take the first match
            row = row.iloc[0]
        fdb_sector = row.get("sector")
        fdb_industry = row.get("industry")
        fdb_cap = row.get("market_cap")

    if fdb_sector is None and fdb_industry is None:
        # Ticker not in financedatabase (common for recent listings/relistings,
        # e.g. SNDK after the 2025 spin-off). yfinance's sector/industry labels
        # use a different taxonomy than financedatabase's, so an exact filter
        # would match nothing — translate them into financedatabase's own
        # vocabulary with a loose match instead.
        info = get_cached_info(ticker)
        if "sector" in df.columns:
            fdb_sector = _loose_taxonomy_match(df["sector"], info.get("sector"))
        if "industry" in df.columns:
            fdb_industry = _loose_taxonomy_match(df["industry"], info.get("industry"))

    def count_excl(frame):
        return sum(1 for s in frame.index if str(s).upper() != ticker)

    # Peer relevance is driven first by SIZE: comparing a mega-cap's multiples to
    # micro-cap medians is meaningless. Build the universe as US-listed names in the
    # same market-cap bucket, then narrow by industry (preferred) or sector.
    universe = df
    if "country" in universe.columns:
        us = universe[universe["country"] == "United States"]
        if not us.empty:
            universe = us
    if fdb_cap and "market_cap" in universe.columns:
        capped = universe[universe["market_cap"] == fdb_cap]
        if not capped.empty:
            universe = capped

    competitors = pd.DataFrame()
    if fdb_industry and "industry" in universe.columns:
        industry_pool = universe[universe["industry"] == fdb_industry]
        if count_excl(industry_pool) >= 4:  # enough same-size industry peers
            competitors = industry_pool
    if competitors.empty and fdb_sector and "sector" in universe.columns:
        competitors = universe[universe["sector"] == fdb_sector]

    # Last resort: if size bucketing left too few, use the sector across all caps.
    if count_excl(competitors) < 4 and fdb_sector and "sector" in df.columns:
        base = df
        if "country" in base.columns:
            us = base[base["country"] == "United States"]
            if not us.empty:
                base = us
        competitors = base[base["sector"] == fdb_sector]

    return {
        "sector": fdb_sector,
        "industry": fdb_industry,
        "market_cap_bucket": fdb_cap,
        "competitors": competitors.head(max_competitors),
    }


# ============================================================
# 6. SEC filings and real risk-factor sources
# ============================================================

def get_sec_cik_map() -> pd.DataFrame:
    """Get SEC ticker-to-CIK mapping."""
    url = "https://www.sec.gov/files/company_tickers.json"
    headers = {"User-Agent": SEC_USER_AGENT}

    response = requests.get(url, headers=headers, timeout=20)
    response.raise_for_status()

    data = response.json()

    df = pd.DataFrame.from_dict(data, orient="index")
    df["ticker"] = df["ticker"].str.upper()
    df["cik_str"] = df["cik_str"].astype(str).str.zfill(10)

    return df


def get_cik_for_ticker(ticker: str) -> Optional[str]:
    mapping = get_sec_cik_map()
    match = mapping[mapping["ticker"] == ticker.upper()]

    if match.empty:
        return None

    return match.iloc[0]["cik_str"]


def retrieve_sec_filings(
    ticker: str,
    forms: Iterable[str] = ("10-K", "10-Q", "8-K", "DEF 14A", "4"),
    limit: int = 20
) -> Dict[str, Any]:
    """
    Retrieve recent SEC filing metadata.

    This does not download full filing text, but it gives accession numbers
    and links needed for deeper filing analysis.
    """
    cik = get_cik_for_ticker(ticker)

    if not cik:
        return {"error": f"Could not find SEC CIK for {ticker}"}

    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    headers = {"User-Agent": SEC_USER_AGENT}

    response = requests.get(url, headers=headers, timeout=20)
    response.raise_for_status()

    payload = response.json()

    recent = payload.get("filings", {}).get("recent", {})
    filings = pd.DataFrame(recent)

    if filings.empty:
        return {
            "cik": cik,
            "filings": filings,
        }

    forms = set(forms)
    filings = filings[filings["form"].isin(forms)].head(limit).copy()

    filings["filing_url"] = filings.apply(
        lambda r: (
            f"https://www.sec.gov/Archives/edgar/data/"
            f"{int(cik)}/{str(r['accessionNumber']).replace('-', '')}/"
            f"{r['primaryDocument']}"
        ),
        axis=1,
    )

    return {
        "cik": cik,
        "company_name": payload.get("name"),
        "sic": payload.get("sic"),
        "sic_description": payload.get("sicDescription"),
        "filings": filings,
    }


def retrieve_risk_factors(ticker: str) -> Dict[str, Any]:
    """
    Lightweight risk scan.

    Uses Yahoo summary plus SEC filing metadata.
    Full 10-K Item 1A parsing would require downloading and parsing filing HTML.
    """
    stock = get_ticker(ticker)

    keywords = [
        "risk", "inflation", "recession", "competition", "regulation",
        "supply", "rates", "litigation", "cybersecurity", "tariff",
        "customer concentration", "supplier", "china", "geopolitical",
    ]

    summary = stock.info.get("longBusinessSummary", "")
    found_keywords = [
        k for k in keywords
        if k.lower() in summary.lower()
    ]

    sec = safe_call(
        lambda: retrieve_sec_filings(ticker, forms=("10-K", "10-Q", "8-K"), limit=10),
        label="SEC filings",
    )

    return {
        "long_summary": summary,
        "summary_keyword_flags": found_keywords,
        "sec_filing_metadata": sec,
    }


# ============================================================
# 7. Analyst expectations, guidance proxies, and revisions
# ============================================================

def retrieve_analyst_expectations(ticker: str) -> Dict[str, Any]:
    stock = get_ticker(ticker)

    return {
        "recommendations": safe_call(
            lambda: stock.recommendations,
            label="recommendations",
        ),
        "upgrades_downgrades": safe_call(
            lambda: stock.upgrades_downgrades,
            label="upgrades_downgrades",
        ),
        "earnings_dates": safe_call(
            lambda: stock.earnings_dates,
            label="earnings_dates",
        ),
        "calendar": safe_call(
            lambda: stock.calendar,
            label="calendar",
        ),
        "analyst_price_targets_from_info": {
            k: get_cached_info(ticker).get(k)
            for k in [
                "targetHighPrice",
                "targetLowPrice",
                "targetMeanPrice",
                "targetMedianPrice",
                "recommendationMean",
                "recommendationKey",
                "numberOfAnalystOpinions",
            ]
        },
    }


# ============================================================
# 8. Peer-relative valuation and quality comparison
# ============================================================

def get_company_snapshot(ticker: str) -> Dict[str, Any]:
    info = get_cached_info(ticker)

    fields = [
        "shortName",
        "sector",
        "industry",
        "marketCap",
        "enterpriseValue",
        "trailingPE",
        "forwardPE",
        "pegRatio",
        "priceToBook",
        "priceToSalesTrailing12Months",
        "enterpriseToRevenue",
        "enterpriseToEbitda",
        "profitMargins",
        "operatingMargins",
        "grossMargins",
        "returnOnAssets",
        "returnOnEquity",
        "revenueGrowth",
        "earningsGrowth",
        "debtToEquity",
        "currentRatio",
        "quickRatio",
        "beta",
        "dividendYield",
        "payoutRatio",
    ]

    return {
        field: info.get(field)
        for field in fields
    }


def retrieve_peer_relative_valuation(
    ticker: str,
    max_peers: int = 10
) -> Dict[str, Any]:
    peers_data = retrieve_industry_competitor_data(ticker, max_competitors=50)
    competitors = peers_data.get("competitors", pd.DataFrame())

    # Find the peer symbols. financedatabase returns the ticker in the DataFrame
    # *index* (not a column), so fall back to the index — without this, peer_tickers
    # was always empty and the peer-median comparison silently never populated.
    raw_symbols = []
    if isinstance(competitors, pd.DataFrame) and not competitors.empty:
        ticker_col = next(
            (c for c in ["symbol", "ticker", "Symbol", "Ticker"] if c in competitors.columns),
            None,
        )
        if ticker_col:
            raw_symbols = competitors[ticker_col].dropna().tolist()
        else:
            raw_symbols = competitors.index.tolist()

    candidates = []
    for sym in raw_symbols:
        sym = str(sym).upper().strip()
        # financedatabase mixes in exchange-suffixed / non-US symbols (e.g. "ABC.L")
        # that yfinance often can't resolve; keep clean US-style tickers only.
        if sym and sym != ticker.upper() and "." not in sym and sym.isalnum():
            candidates.append(sym)
        # Examine a few more than we need so we can drop invalid ones below.
        if len(candidates) >= max_peers * 2:
            break

    # Fetch each candidate's snapshot and keep only valid common-stock peers — ones
    # with a real market cap. This drops preferred shares / warrants (e.g. "BACRP",
    # "OSTBP") and delisted/unresolvable tickers that financedatabase mixes in, which
    # would otherwise show up as junk "peers" (their NaNs don't move the medians, but
    # they look wrong in the list). Stop once we have enough.
    snapshots = {ticker.upper(): get_company_snapshot(ticker)}
    peer_tickers = []
    for sym in candidates:
        if len(peer_tickers) >= max_peers:
            break
        snap = safe_call(lambda p=sym: get_company_snapshot(p), label=f"peer_snapshot:{sym}")
        if isinstance(snap, dict) and clean_number(snap.get("marketCap")) is not None:
            snapshots[sym] = snap
            peer_tickers.append(sym)

    comparison = pd.DataFrame.from_dict(snapshots, orient="index")

    numeric_cols = comparison.select_dtypes(include=[np.number]).columns.tolist()

    if numeric_cols:
        peer_medians = comparison.loc[
            comparison.index != ticker.upper(),
            numeric_cols
        ].median(numeric_only=True)
    else:
        peer_medians = pd.Series(dtype=float)

    return {
        "target_snapshot": snapshots[ticker.upper()],
        "peer_tickers_used": peer_tickers,
        "comparison_table": comparison,
        "peer_medians": peer_medians,
    }


# ============================================================
# 9. Capital allocation and earnings quality
# ============================================================

def analyze_capital_allocation(ticker: str) -> Dict[str, Any]:
    # All flow figures are trailing-twelve-month (see ttm_value) so buybacks,
    # dividends, capex, and SBC reflect the last 4 reported quarters — matching
    # the freeCashflow field from .info, which Yahoo also reports as TTM.
    stock = get_ticker(ticker)

    cashflow = stock.cashflow
    income = stock.income_stmt
    balance = stock.balance_sheet
    info = get_cached_info(ticker)
    q_cashflow = stock.quarterly_cashflow
    q_income = stock.quarterly_financials
    q_balance = stock.quarterly_balance_sheet

    buybacks = ttm_value(q_cashflow, cashflow, "Repurchase Of Capital Stock")
    dividends_paid = ttm_value(q_cashflow, cashflow, "Cash Dividends Paid")
    capex = ttm_value(q_cashflow, cashflow, "Capital Expenditure")
    free_cash_flow = info.get("freeCashflow")
    operating_cash_flow = ttm_value(q_cashflow, cashflow, "Operating Cash Flow")
    net_income = ttm_value(q_income, income, "Net Income")
    stock_based_comp = ttm_value(q_cashflow, cashflow, "Stock Based Compensation")
    revenue = ttm_value(q_income, income, "Total Revenue")

    return {
        "free_cash_flow_from_info": free_cash_flow,
        "operating_cash_flow_latest": operating_cash_flow,
        "net_income_latest": net_income,
        "cash_conversion_ocf_to_net_income": (
            operating_cash_flow / net_income
            if operating_cash_flow is not None and net_income not in (None, 0)
            else None
        ),
        "capital_expenditure_latest": capex,
        "buybacks_latest": buybacks,
        "dividends_paid_latest": dividends_paid,
        "stock_based_compensation_latest": stock_based_comp,
        "sbc_as_percent_of_revenue": (
            stock_based_comp / revenue
            if stock_based_comp is not None and revenue not in (None, 0)
            else None
        ),
        "retained_earnings_latest": mrq_value(q_balance, balance, "Retained Earnings"),
    }


def analyze_earnings_quality(ticker: str) -> Dict[str, Any]:
    # Flow figures are trailing-twelve-month (sum of the last 4 quarters) so the
    # metrics track the latest earnings report instead of the last fiscal year,
    # which can be ~a year stale. Balance-sheet items are most-recent-quarter.
    stock = get_ticker(ticker)

    cashflow = stock.cashflow
    income = stock.income_stmt
    balance = stock.balance_sheet
    q_cashflow = stock.quarterly_cashflow
    q_income = stock.quarterly_financials
    q_balance = stock.quarterly_balance_sheet

    revenue = ttm_value(q_income, income, "Total Revenue")
    net_income = ttm_value(q_income, income, "Net Income")
    ocf = ttm_value(q_cashflow, cashflow, "Operating Cash Flow")
    fcf = ttm_value(q_cashflow, cashflow, "Free Cash Flow")
    receivables = mrq_value(q_balance, balance, "Accounts Receivable")
    inventory = mrq_value(q_balance, balance, "Inventory")

    return {
        "revenue_latest": revenue,
        "net_income_latest": net_income,
        "operating_cash_flow_latest": ocf,
        "free_cash_flow_latest": fcf,
        "ocf_minus_net_income": (
            ocf - net_income
            if ocf is not None and net_income is not None
            else None
        ),
        "fcf_margin": (
            fcf / revenue
            if fcf is not None and revenue not in (None, 0)
            else None
        ),
        "net_margin_calculated": (
            net_income / revenue
            if net_income is not None and revenue not in (None, 0)
            else None
        ),
        "accounts_receivable_latest": receivables,
        "inventory_latest": inventory,
        "red_flags_to_review": [
            flag
            for flag, triggered in {
                "Operating cash flow is below net income": (
                    ocf is not None and net_income is not None and ocf < net_income
                ),
                "Free cash flow is negative": (
                    fcf is not None and fcf < 0
                ),
                "Inventory exists and should be checked for buildup": (
                    inventory is not None and inventory > 0
                ),
            }.items()
            if triggered
        ],
    }


# ============================================================
# 10. Debt, liquidity, and balance-sheet risk
# ============================================================

def analyze_balance_sheet_risk(ticker: str) -> Dict[str, Any]:
    stock = get_ticker(ticker)

    info = get_cached_info(ticker)
    balance = stock.balance_sheet
    income = stock.income_stmt
    q_balance = stock.quarterly_balance_sheet
    q_income = stock.quarterly_financials

    # Debt and cash come from .info (most recent quarter) with a statement
    # fallback; interest coverage uses TTM flows so it reflects the last four
    # reported quarters rather than the last fiscal year.
    total_debt = info.get("totalDebt") or mrq_value(q_balance, balance, "Total Debt")
    cash = info.get("totalCash") or mrq_value(q_balance, balance, "Cash And Cash Equivalents")
    ebitda = info.get("ebitda")

    operating_income = ttm_value(q_income, income, "Operating Income")
    interest_expense = ttm_value(q_income, income, "Interest Expense")

    return {
        "total_debt": total_debt,
        "cash": cash,
        "net_debt": (
            total_debt - cash
            if total_debt is not None and cash is not None
            else None
        ),
        "ebitda": ebitda,
        "net_debt_to_ebitda": (
            (total_debt - cash) / ebitda
            if total_debt is not None and cash is not None and ebitda not in (None, 0)
            else None
        ),
        "debt_to_equity_from_info": info.get("debtToEquity"),
        "current_ratio": info.get("currentRatio"),
        "quick_ratio": info.get("quickRatio"),
        "interest_expense": interest_expense,
        "operating_income": operating_income,
        "interest_coverage": (
            operating_income / abs(interest_expense)
            if operating_income is not None and interest_expense not in (None, 0)
            else None
        ),
    }


# ============================================================
# 11. Insider transactions, ownership quality, and holders
# ============================================================

def retrieve_ownership_and_insider_data(ticker: str) -> Dict[str, Any]:
    stock = get_ticker(ticker)

    return {
        "major_holders": safe_call(
            lambda: stock.major_holders,
            label="major_holders",
        ),
        "institutional_holders": safe_call(
            lambda: stock.institutional_holders,
            label="institutional_holders",
        ),
        "mutualfund_holders": safe_call(
            lambda: stock.mutualfund_holders,
            label="mutualfund_holders",
        ),
        "insider_transactions": safe_call(
            lambda: stock.insider_transactions,
            label="insider_transactions",
        ),
        "insider_purchases": safe_call(
            lambda: stock.insider_purchases,
            label="insider_purchases",
        ),
        "insider_roster_holders": safe_call(
            lambda: stock.insider_roster_holders,
            label="insider_roster_holders",
        ),
    }


# ============================================================
# 12. Options, implied volatility, and market expectations
# ============================================================

def retrieve_options_market_data(
    ticker: str,
    max_expirations: int = 4
) -> Dict[str, Any]:
    stock = get_ticker(ticker)

    expirations = list(getattr(stock, "options", []) or [])[:max_expirations]
    chains = {}

    for exp in expirations:
        chains[exp] = safe_call(
            lambda e=exp: stock.option_chain(e),
            label=f"option_chain:{exp}",
        )

    summary = []

    for exp, chain in chains.items():
        if isinstance(chain, dict) and "error" in chain:
            continue

        calls = getattr(chain, "calls", pd.DataFrame())
        puts = getattr(chain, "puts", pd.DataFrame())

        call_volume = (
            calls.get("volume", pd.Series(dtype=float)).sum()
            if not calls.empty
            else 0
        )

        put_volume = (
            puts.get("volume", pd.Series(dtype=float)).sum()
            if not puts.empty
            else 0
        )

        call_oi = (
            calls.get("openInterest", pd.Series(dtype=float)).sum()
            if not calls.empty
            else 0
        )

        put_oi = (
            puts.get("openInterest", pd.Series(dtype=float)).sum()
            if not puts.empty
            else 0
        )

        summary.append({
            "expiration": exp,
            "call_volume": call_volume,
            "put_volume": put_volume,
            "put_call_volume_ratio": (
                put_volume / call_volume
                if call_volume
                else None
            ),
            "call_open_interest": call_oi,
            "put_open_interest": put_oi,
            "put_call_open_interest_ratio": (
                put_oi / call_oi
                if call_oi
                else None
            ),
        })

    return {
        "expirations_checked": expirations,
        "option_chain_summary": pd.DataFrame(summary),
        "raw_chains": chains,
    }


# ============================================================
# 13. Short interest and bearish positioning proxies
# ============================================================

def retrieve_short_interest_data(ticker: str) -> Dict[str, Any]:
    info = get_cached_info(ticker)

    fields = [
        "shortRatio",
        "shortPercentOfFloat",
        "sharesShort",
        "sharesShortPriorMonth",
        "sharesPercentSharesOut",
        "floatShares",
        "heldPercentInsiders",
        "heldPercentInstitutions",
    ]

    return {
        field: info.get(field)
        for field in fields
    }


# ============================================================
# 14. Industry-specific KPI framework
# ============================================================

def identify_industry_kpis(ticker: str) -> Dict[str, Any]:
    """
    Returns which operating KPIs should be researched for this company.

    Most of these are not standardized in Yahoo Finance and require filings,
    earnings decks, or specialized datasets.
    """
    info = get_cached_info(ticker)

    sector = str(info.get("sector", "")).lower()
    industry = str(info.get("industry", "")).lower()
    text = f"{sector} {industry}"

    kpi_map = {
        "software_saas": [
            "ARR",
            "net revenue retention",
            "gross retention",
            "CAC",
            "LTV",
            "churn",
            "RPO",
            "billings",
        ],
        "banks": [
            "net interest margin",
            "deposit growth",
            "loan growth",
            "charge-offs",
            "CET1",
            "tangible book value",
        ],
        "retail": [
            "same-store sales",
            "traffic",
            "basket size",
            "inventory turnover",
            "gross margin",
            "store count",
        ],
        "semiconductors": [
            "backlog",
            "inventory days",
            "wafer starts",
            "utilization",
            "ASP",
            "end-market mix",
        ],
        "energy": [
            "production volume",
            "proved reserves",
            "realized price",
            "lifting cost",
            "reserve replacement",
        ],
        "airlines": [
            "load factor",
            "ASM",
            "RASM",
            "CASM",
            "fuel cost",
            "fleet utilization",
        ],
        "insurance": [
            "combined ratio",
            "loss ratio",
            "expense ratio",
            "float",
            "book value growth",
        ],
        "biotech_pharma": [
            "pipeline stage",
            "trial readouts",
            "FDA calendar",
            "patent cliff",
            "R&D productivity",
        ],
        "reit": [
            "FFO",
            "AFFO",
            "occupancy",
            "lease spreads",
            "same-store NOI",
            "debt maturity schedule",
        ],
        "media_internet": [
            "DAU",
            "MAU",
            "ARPU",
            "engagement",
            "ad impressions",
            "take rate",
        ],
    }

    suggested = []

    if any(w in text for w in ["software", "cloud", "application"]):
        suggested += kpi_map["software_saas"]

    if "bank" in text or "financial" in text:
        suggested += kpi_map["banks"]

    if any(w in text for w in ["retail", "apparel", "restaurants"]):
        suggested += kpi_map["retail"]

    if "semiconductor" in text:
        suggested += kpi_map["semiconductors"]

    if any(w in text for w in ["oil", "gas", "energy"]):
        suggested += kpi_map["energy"]

    if "airline" in text:
        suggested += kpi_map["airlines"]

    if "insurance" in text:
        suggested += kpi_map["insurance"]

    if any(w in text for w in ["biotech", "pharmaceutical", "drug"]):
        suggested += kpi_map["biotech_pharma"]

    if "reit" in text or "real estate" in text:
        suggested += kpi_map["reit"]

    if any(w in text for w in ["internet", "interactive media", "entertainment"]):
        suggested += kpi_map["media_internet"]

    return {
        "sector": info.get("sector"),
        "industry": info.get("industry"),
        "suggested_kpis_to_research": sorted(set(suggested)),
        "note": (
            "These KPIs usually require SEC filings, investor presentations, "
            "earnings transcripts, or specialized datasets."
        ),
    }


# ============================================================
# 15. Management quality and governance proxies
# ============================================================

def retrieve_governance_data(ticker: str) -> Dict[str, Any]:
    stock = get_ticker(ticker)
    info = get_cached_info(ticker)

    governance_fields = [
        "auditRisk",
        "boardRisk",
        "compensationRisk",
        "shareHolderRightsRisk",
        "overallRisk",
        "governanceEpochDate",
        "compensationAsOfEpochDate",
    ]

    return {
        "governance_risk_fields": {
            k: info.get(k)
            for k in governance_fields
        },
        "company_officers": info.get("companyOfficers"),
        "proxy_filings": safe_call(
            lambda: retrieve_sec_filings(ticker, forms=("DEF 14A",), limit=5),
            label="proxy_filings",
        ),
    }


# ============================================================
# 16. Geographic/currency exposure and regulatory placeholders
# ============================================================

def retrieve_exposure_checklist(ticker: str) -> Dict[str, Any]:
    info = get_ticker(ticker).info
    summary = str(info.get("longBusinessSummary", "")).lower()

    exposure_terms = [
        "china",
        "europe",
        "international",
        "foreign exchange",
        "currency",
        "tariff",
        "export",
        "import",
        "sanctions",
        "regulation",
        "fda",
        "antitrust",
        "privacy",
        "labor",
        "union",
        "environmental",
    ]

    return {
        "detected_exposure_terms_in_summary": [
            term for term in exposure_terms
            if term in summary
        ],
        "note": (
            "For precise geographic revenue and regulatory exposure, "
            "parse 10-K segment notes and risk factors."
        ),
    }


# ============================================================
# 17. Alternative data placeholders
# ============================================================

def alternative_data_checklist(ticker: str) -> Dict[str, Any]:
    return {
        "ticker": ticker.upper(),
        "not_collected_without_special_sources": [
            "web traffic",
            "app downloads",
            "Google Trends",
            "credit/debit card spend",
            "job postings",
            "employee reviews",
            "shipping/import records",
            "satellite data",
            "customer reviews",
            "social media activity",
            "developer activity",
            "patent filings",
            "store foot traffic",
            "product pricing changes",
        ],
        "how_to_add": {
            "Google Trends": "Use pytrends or a trends API.",
            "job postings": "Use company career pages or a jobs dataset.",
            "app downloads": (
                "Use Sensor Tower, data.ai, Similarweb, or another "
                "app-intelligence source."
            ),
            "web traffic": (
                "Use Similarweb, Semrush, Ahrefs, or first-party analytics "
                "if available."
            ),
            "patents": "Use USPTO, Google Patents, or a patents API.",
        },
    }


# ============================================================
# 18. Historical cycle and factor analysis
# ============================================================

def analyze_historical_performance(ticker: str) -> Dict[str, Any]:
    prices = yf.download(ticker, period="max", auto_adjust=True, progress=False)
    prices = flatten_yfinance_columns(prices)

    close_col = "Close" if "Close" in prices.columns else next(
        (c for c in prices.columns if c.startswith("Close_")),
        None,
    )

    if close_col is None or prices.empty:
        return {"error": "No price history available"}

    close = prices[close_col].dropna()
    returns = close.pct_change().dropna()

    years = (close.index[-1] - close.index[0]).days / 365.25

    cagr = (
        (close.iloc[-1] / close.iloc[0]) ** (1 / years) - 1
        if years > 0
        else None
    )

    max_drawdown = (close / close.cummax() - 1).min()
    annualized_vol = returns.std() * np.sqrt(252)

    sharpe_like = (
        cagr / annualized_vol
        if cagr is not None and annualized_vol
        else None
    )

    return {
        "start_date": str(close.index[0].date()),
        "end_date": str(close.index[-1].date()),
        "total_return": close.iloc[-1] / close.iloc[0] - 1,
        "cagr": cagr,
        "annualized_volatility": annualized_vol,
        "max_drawdown": max_drawdown,
        "return_to_volatility_ratio": sharpe_like,
    }


# ============================================================
# 19. Portfolio context helper
# ============================================================

def analyze_portfolio_context(
    ticker: str,
    current_holdings: Optional[List[str]] = None
) -> Dict[str, Any]:
    if not current_holdings:
        return {
            "note": (
                "No portfolio holdings supplied. "
                "Pass current_holdings=['AAPL','MSFT',...] to calculate correlations."
            ),
        }

    tickers = sorted(set([ticker.upper()] + [h.upper() for h in current_holdings]))

    prices = yf.download(tickers, period="3y", auto_adjust=True, progress=False)["Close"]
    returns = prices.pct_change().dropna()
    corr = returns.corr()

    return {
        "tickers_analyzed": tickers,
        "correlation_matrix": corr,
        "target_correlation_to_holdings": corr[ticker.upper()].drop(
            labels=[ticker.upper()],
            errors="ignore",
        ),
    }


# ============================================================
# 20. Rule-based stock edge and moat/niche scoring
# ============================================================

def assess_stock_edge(ticker: str) -> Dict[str, Any]:
    quant = retrieve_quantitative_data(ticker)
    tech = retrieve_technical_indicators(ticker)

    price = quant["price_history"].get("Close", pd.Series(dtype=float)).dropna()

    close_col = "Close" if "Close" in tech.columns else next(
        (c for c in tech.columns if c.startswith("Close_")),
        None,
    )

    if price.empty or close_col is None or tech.empty:
        return {"error": "Not enough price data to assess edge"}

    return {
        "uptrend_above_200d": bool(
            price.iloc[-1] > price.rolling(200).mean().iloc[-1]
        ),
        "above_SMA_50": bool(
            tech[close_col].iloc[-1] > tech["SMA_50"].iloc[-1]
        ),
        "positive_MACD": bool(
            tech["MACD"].iloc[-1] > tech["Signal"].iloc[-1]
        ),
        "current_volatility_20d_annualized": clean_number(
            tech["Volatility_20d_annualized"].iloc[-1]
        ),
        "current_RSI": clean_number(
            tech["RSI"].iloc[-1]
        ),
        "current_drawdown": clean_number(
            tech["Drawdown"].iloc[-1]
        ),
    }


def check_stock_niche(ticker: str) -> Dict[str, Any]:
    info = get_ticker(ticker).info
    info_text = str(info).lower()

    return {
        "brand_word_present": "brand" in info_text,
        "patents_from_info_if_available": info.get("numberOfPatents", 0),
        "market_cap_proxy_for_scale": info.get("marketCap", 0),
        "unique_product_terms_present": any(
            word in info_text
            for word in ["ecosystem", "unique", "monopoly", "proprietary"]
        ),
        "switching_cost_terms_present": any(
            word in info_text
            for word in ["subscription", "platform", "integrated", "mission-critical"]
        ),
        "network_effect_terms_present": any(
            word in info_text
            for word in ["network", "marketplace", "community", "users"]
        ),
        "note": (
            "This is only a text/scale proxy. Real moat analysis requires "
            "product, customer, and competitive research."
        ),
    }


# ============================================================
# 21. Full assessment
# ============================================================

def comprehensive_stock_assessment(
    ticker: str,
    include_macro: bool = True,
    include_sec: bool = True,
    include_options: bool = True,
    current_holdings: Optional[List[str]] = None,
    lean: bool = False,
) -> Dict[str, Any]:
    ticker = ticker.upper().strip()

    # Start from a clean cache so each assessment fetches fresh data, then warm up
    # the yfinance session (with retries) so the cold first-call handshake doesn't
    # make the first load fail.
    _TICKER_CACHE.pop(ticker, None)
    prime_ticker(ticker)

    # Each data source below is an independent, network-bound call. They run
    # sequentially on purpose: most of them hit Yahoo through one shared yfinance
    # session, and firing them concurrently makes Yahoo return intermittent 401s
    # (the crumb/cookie handshake is not concurrency-safe) — which is exactly the
    # flakiness we just removed. They are cheap anyway (~26s combined); the real
    # cost was the macro/FRED source, which is parallelized internally where it is
    # safe to do so (FRED is a separate, robust backend). See retrieve_macro_data.
    # Sources consumed by the scored summary / web output.
    tasks = {
        "quantitative": lambda: retrieve_quantitative_data(ticker),
        # NB: industry/competitor data is intentionally not fetched as its own
        # source here. retrieve_peer_relative_valuation() already calls
        # retrieve_industry_competitor_data() internally to build the peer set, so a
        # standalone fetch was pure duplication (and ~7s of it). The valuable output
        # of that data — the peer-median comparison — is surfaced via build_peer_comparison().
        "analyst_expectations": lambda: retrieve_analyst_expectations(ticker),
        "peer_relative_valuation": lambda: retrieve_peer_relative_valuation(ticker),
        "capital_allocation": lambda: analyze_capital_allocation(ticker),
        "earnings_quality": lambda: analyze_earnings_quality(ticker),
        "balance_sheet_risk": lambda: analyze_balance_sheet_risk(ticker),
        "short_interest": lambda: retrieve_short_interest_data(ticker),
        "industry_kpi_framework": lambda: identify_industry_kpis(ticker),
        "governance": lambda: retrieve_governance_data(ticker),
        "historical_performance": lambda: analyze_historical_performance(ticker),
        "edge": lambda: assess_stock_edge(ticker),
    }

    # Sources fetched for the CLI's full report but consumed by nothing in the
    # scored summary or web JSON. The web path (lean=True) skips them — they were
    # costing ~10-20s per request with zero effect on the displayed result.
    # If one of these gets wired into the output later, move it up into `tasks`.
    if not lean:
        tasks.update({
            "news_and_sentiment": lambda: retrieve_basic_news_sentiment(ticker),
            "technical": lambda: retrieve_technical_indicators(ticker),
            "broad_market": retrieve_broad_market_data,
            "risk_factors": lambda: retrieve_risk_factors(ticker),
            "ownership_and_insiders": lambda: retrieve_ownership_and_insider_data(ticker),
            "geographic_regulatory_exposure": lambda: retrieve_exposure_checklist(ticker),
            "portfolio_context": lambda: analyze_portfolio_context(ticker, current_holdings),
            "niche": lambda: check_stock_niche(ticker),
        })

    if include_macro:
        tasks["macro"] = retrieve_macro_data
    if include_sec:
        tasks["sec_filings"] = lambda: retrieve_sec_filings(ticker)
    if include_options:
        tasks["options_market"] = lambda: retrieve_options_market_data(ticker)

    # alternative_data_checklist is a local (non-network) call, so keep it inline.
    assessment: Dict[str, Any] = {
        "ticker": ticker,
        "alternative_data_checklist": alternative_data_checklist(ticker),
    }

    for key, fn in tasks.items():
        assessment[key] = safe_call(fn, label=key)

    # Macro regime is cheap (a few fast FRED series, cached for hours) and feeds
    # the per-stock macro risk overlay, so include it on every path — including the
    # web path that skips the heavy raw `macro` source above.
    assessment["macro_regime"] = safe_call(get_macro_regime, label="macro_regime")

    return assessment


# ============================================================
# 22. Readable console output
# ============================================================

def print_section(title: str, value: Any, max_rows: int = 8):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

    if isinstance(value, pd.DataFrame):
        print(value.tail(max_rows) if len(value) > max_rows else value)

    elif isinstance(value, pd.Series):
        print(value.tail(max_rows) if len(value) > max_rows else value)

    elif isinstance(value, dict):
        for k, v in value.items():
            if isinstance(v, pd.DataFrame):
                print(f"\n{k}:")
                print(v.head(max_rows))

            elif isinstance(v, pd.Series):
                print(f"\n{k}:")
                print(v.head(max_rows))

            elif isinstance(v, dict):
                print(f"\n{k}:")
                printable = {
                    sub_k: sub_v
                    for sub_k, sub_v in v.items()
                    if not isinstance(sub_v, (pd.DataFrame, pd.Series))
                }
                print(json.dumps(printable, indent=2, default=str)[:3000])

                for sub_k, sub_v in v.items():
                    if isinstance(sub_v, (pd.DataFrame, pd.Series)):
                        print(f"\n{k}.{sub_k}:")
                        print(sub_v.head(max_rows))

            else:
                text = str(v)
                print(f"{k}: {text[:1500]}{'...' if len(text) > 1500 else ''}")

    else:
        print(value)


# def summarize_assessment(assessment: Dict[str, Any]):
#     """Print a digest instead of dumping giant nested objects."""
#     print_section("Ticker", assessment.get("ticker"))
#     print_section("Edge", assessment.get("edge"))
#     print_section("Niche / Moat Signals", assessment.get("niche"))
#     print_section("Balance Sheet Risk", assessment.get("balance_sheet_risk"))
#     print_section("Earnings Quality", assessment.get("earnings_quality"))
#     print_section("Capital Allocation", assessment.get("capital_allocation"))
#     print_section("Short Interest", assessment.get("short_interest"))
#     print_section("Industry KPI Framework", assessment.get("industry_kpi_framework"))
#     print_section("Analyst Expectations", assessment.get("analyst_expectations"))
#     print_section("Peer Relative Valuation", assessment.get("peer_relative_valuation"))
#     print_section("Risk Factors", assessment.get("risk_factors"))
#     print_section("Governance", assessment.get("governance"))
#     print_section("Historical Performance", assessment.get("historical_performance"))
#     print_section("Options Market", assessment.get("options_market"))
#     print_section("Alternative Data Checklist", assessment.get("alternative_data_checklist"))

# ============================================================
# 22. Concise scored report output
# ============================================================

def clamp(value, low=0, high=10):
    if value is None:
        return None
    return max(low, min(high, value))


def pct(value):
    if value is None:
        return "N/A"
    try:
        return f"{value * 100:.1f}%"
    except Exception:
        return "N/A"


def num(value, digits=2):
    if value is None:
        return "N/A"
    try:
        return f"{float(value):,.{digits}f}"
    except Exception:
        return "N/A"


def money(value):
    if value is None:
        return "N/A"
    try:
        value = float(value)
        abs_value = abs(value)

        if abs_value >= 1_000_000_000_000:
            return f"${value / 1_000_000_000_000:.2f}T"
        if abs_value >= 1_000_000_000:
            return f"${value / 1_000_000_000:.2f}B"
        if abs_value >= 1_000_000:
            return f"${value / 1_000_000:.2f}M"

        return f"${value:,.0f}"
    except Exception:
        return "N/A"


def get_nested(dct, *keys, default=None):
    cur = dct
    for key in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(key)
    return cur if cur is not None else default


def score_balance_sheet(balance):
    """
    Higher score = better balance sheet.
    """
    if not isinstance(balance, dict) or "error" in balance:
        return 5, ["Balance-sheet data unavailable or incomplete."]

    score = 7
    notes = []

    net_debt_to_ebitda = balance.get("net_debt_to_ebitda")
    current_ratio = balance.get("current_ratio")
    interest_coverage = balance.get("interest_coverage")
    net_debt = balance.get("net_debt")

    if net_debt is not None and net_debt < 0:
        score += 1
        notes.append("Net cash position supports financial flexibility.")
    elif net_debt_to_ebitda is not None:
        if net_debt_to_ebitda > 4:
            score -= 3
            notes.append("High net debt to EBITDA increases balance-sheet risk.")
        elif net_debt_to_ebitda > 2.5:
            score -= 2
            notes.append("Moderate leverage deserves monitoring.")
        elif net_debt_to_ebitda < 1:
            score += 1
            notes.append("Leverage appears low relative to EBITDA.")

    if current_ratio is not None:
        if current_ratio < 1:
            score -= 2
            notes.append("Current ratio below 1 may indicate liquidity pressure.")
        elif current_ratio > 1.5:
            score += 1
            notes.append("Current ratio suggests adequate short-term liquidity.")

    if interest_coverage is not None:
        if interest_coverage < 2:
            score -= 3
            notes.append("Weak interest coverage is a major concern.")
        elif interest_coverage < 5:
            score -= 1
            notes.append("Interest coverage is acceptable but not especially strong.")
        elif interest_coverage > 8:
            score += 1
            notes.append("Strong interest coverage reduces credit risk.")

    return clamp(score), notes or ["No major balance-sheet red flags detected."]


def score_earnings_quality(earnings):
    """
    Higher score = better earnings quality.
    """
    if not isinstance(earnings, dict) or "error" in earnings:
        return 5, ["Earnings-quality data unavailable or incomplete."]

    score = 7
    notes = []

    fcf_margin = earnings.get("fcf_margin")
    net_margin = earnings.get("net_margin_calculated")
    ocf_minus_net_income = earnings.get("ocf_minus_net_income")
    red_flags = earnings.get("red_flags_to_review", [])

    if fcf_margin is not None:
        if fcf_margin < 0:
            score -= 3
            notes.append("Free cash flow margin is negative.")
        elif fcf_margin < 0.05:
            score -= 1
            notes.append("Free cash flow margin is thin.")
        elif fcf_margin > 0.15:
            score += 1
            notes.append("Free cash flow margin is healthy.")

    if net_margin is not None:
        if net_margin < 0:
            score -= 2
            notes.append("Net margin is negative.")
        elif net_margin > 0.15:
            score += 1
            notes.append("Net margin is strong.")

    if ocf_minus_net_income is not None:
        if ocf_minus_net_income < 0:
            score -= 2
            notes.append("Operating cash flow is below net income.")
        else:
            score += 1
            notes.append("Operating cash flow supports reported earnings.")

    if red_flags:
        score -= min(len(red_flags), 3)
        notes.extend(red_flags[:3])

    return clamp(score), notes or ["Earnings quality appears acceptable from available data."]


def score_valuation(snapshot, peer_medians):
    """
    Higher score = more attractive valuation.
    """
    if not isinstance(snapshot, dict) or "error" in snapshot:
        return 5, ["Valuation data unavailable or incomplete."]

    score = 5
    notes = []

    forward_pe = snapshot.get("forwardPE")
    trailing_pe = snapshot.get("trailingPE")
    ev_ebitda = snapshot.get("enterpriseToEbitda")
    ps = snapshot.get("priceToSalesTrailing12Months")
    peg = snapshot.get("pegRatio")

    peer_forward_pe = None
    peer_ev_ebitda = None
    peer_ps = None

    try:
        peer_forward_pe = clean_number(peer_medians.get("forwardPE"))
        peer_ev_ebitda = clean_number(peer_medians.get("enterpriseToEbitda"))
        peer_ps = clean_number(peer_medians.get("priceToSalesTrailing12Months"))
    except Exception:
        pass

    if forward_pe is not None and peer_forward_pe is not None and peer_forward_pe > 0:
        if forward_pe < peer_forward_pe * 0.8:
            score += 2
            notes.append("Forward P/E is meaningfully below peer median.")
        elif forward_pe > peer_forward_pe * 1.3:
            score -= 2
            notes.append("Forward P/E is meaningfully above peer median.")

    elif forward_pe is not None:
        if forward_pe < 15:
            score += 1
            notes.append("Forward P/E looks inexpensive on an absolute basis.")
        elif forward_pe > 35:
            score -= 1
            notes.append("Forward P/E looks expensive on an absolute basis.")

    if ev_ebitda is not None and peer_ev_ebitda is not None and peer_ev_ebitda > 0:
        if ev_ebitda < peer_ev_ebitda * 0.8:
            score += 1
            notes.append("EV/EBITDA is below peer median.")
        elif ev_ebitda > peer_ev_ebitda * 1.3:
            score -= 1
            notes.append("EV/EBITDA is above peer median.")

    if ps is not None and peer_ps is not None and peer_ps > 0:
        if ps > peer_ps * 1.5:
            score -= 1
            notes.append("Price/sales is elevated versus peers.")

    if peg is not None:
        if peg > 2:
            score -= 1
            notes.append("PEG ratio suggests valuation may be high relative to growth.")
        elif 0 < peg < 1.2:
            score += 1
            notes.append("PEG ratio appears reasonable relative to growth.")

    if trailing_pe is not None and trailing_pe < 0:
        score -= 1
        notes.append("Negative trailing P/E indicates current unprofitability.")

    return clamp(score), notes or ["Valuation appears roughly neutral from available metrics."]


def score_growth_quality(snapshot):
    """
    Higher score = better growth/profitability profile.
    """
    if not isinstance(snapshot, dict) or "error" in snapshot:
        return 5, ["Growth/profitability data unavailable or incomplete."]

    score = 5
    notes = []

    revenue_growth = snapshot.get("revenueGrowth")
    earnings_growth = snapshot.get("earningsGrowth")
    operating_margin = snapshot.get("operatingMargins")
    gross_margin = snapshot.get("grossMargins")
    roe = snapshot.get("returnOnEquity")

    if revenue_growth is not None:
        if revenue_growth > 0.15:
            score += 2
            notes.append("Revenue growth is strong.")
        elif revenue_growth > 0.05:
            score += 1
            notes.append("Revenue growth is positive.")
        elif revenue_growth < 0:
            score -= 2
            notes.append("Revenue is declining.")

    if earnings_growth is not None:
        if earnings_growth > 0.15:
            score += 1
            notes.append("Earnings growth is strong.")
        elif earnings_growth < 0:
            score -= 1
            notes.append("Earnings are declining.")

    if operating_margin is not None:
        if operating_margin > 0.20:
            score += 1
            notes.append("Operating margin is strong.")
        elif operating_margin < 0:
            score -= 2
            notes.append("Operating margin is negative.")

    if gross_margin is not None and gross_margin > 0.50:
        score += 1
        notes.append("Gross margin suggests strong pricing power or software-like economics.")

    if roe is not None:
        if roe > 0.15:
            score += 1
            notes.append("Return on equity is attractive.")
        elif roe < 0:
            score -= 1
            notes.append("Return on equity is negative.")

    return clamp(score), notes or ["Growth and profitability appear mixed or incomplete."]


def score_market_technical(edge, historical, short_interest):
    """
    Higher score = better market/technical setup.
    """
    score = 5
    notes = []

    if isinstance(edge, dict) and "error" not in edge:
        if edge.get("uptrend_above_200d"):
            score += 1
            notes.append("Price is above its 200-day average.")
        else:
            score -= 1
            notes.append("Price is below its 200-day average.")

        if edge.get("positive_MACD"):
            score += 1
            notes.append("MACD trend is positive.")
        else:
            score -= 1
            notes.append("MACD trend is not positive.")

        rsi = edge.get("current_RSI")
        if rsi is not None:
            if rsi > 75:
                score -= 1
                notes.append("RSI is elevated; near-term overbought risk.")
            elif rsi < 30:
                score -= 1
                notes.append("RSI is depressed; momentum remains weak.")

        drawdown = edge.get("current_drawdown")
        if drawdown is not None and drawdown < -0.30:
            score -= 1
            notes.append("Stock remains in a large drawdown.")

    if isinstance(historical, dict) and "error" not in historical:
        max_drawdown = historical.get("max_drawdown")
        vol = historical.get("annualized_volatility")

        if max_drawdown is not None and max_drawdown < -0.60:
            score -= 1
            notes.append("Historical max drawdown has been severe.")

        if vol is not None:
            if vol > 0.55:
                score -= 2
                notes.append("Historical volatility is very high.")
            elif vol > 0.35:
                score -= 1
                notes.append("Historical volatility is elevated.")

    if isinstance(short_interest, dict):
        short_float = short_interest.get("shortPercentOfFloat")
        short_ratio = short_interest.get("shortRatio")

        if short_float is not None and short_float > 0.10:
            score -= 1
            notes.append("Short interest is elevated.")
        if short_ratio is not None and short_ratio > 7:
            score -= 1
            notes.append("Days-to-cover short ratio is high.")

    return clamp(score), notes or ["Technical and market-positioning picture is neutral."]


def score_governance(governance):
    """
    Higher score = better governance.
    Yahoo governance fields are usually 1-10 where higher often means higher risk,
    so this converts them into an approximate quality score.
    """
    if not isinstance(governance, dict) or "error" in governance:
        return 5, ["Governance data unavailable or incomplete."]

    fields = governance.get("governance_risk_fields", {})

    risk_values = [
        fields.get("auditRisk"),
        fields.get("boardRisk"),
        fields.get("compensationRisk"),
        fields.get("shareHolderRightsRisk"),
        fields.get("overallRisk"),
    ]

    risk_values = [clean_number(x) for x in risk_values if clean_number(x) is not None]

    if not risk_values:
        return 5, ["No governance risk scores available."]

    avg_risk = sum(risk_values) / len(risk_values)
    score = 11 - avg_risk

    notes = [f"Average available governance risk score is {avg_risk:.1f}/10."]

    if avg_risk >= 7:
        notes.append("Governance risk appears elevated.")
    elif avg_risk <= 3:
        notes.append("Governance risk appears relatively low.")

    return clamp(score), notes


# Plain-English meaning for each metric label, shown beside the number so the
# abbreviations are never bare. Keys match the labels in build_metric_snapshot /
# build_peer_comparison.
METRIC_GLOSSARY = {
    "Forward P/E": "Price versus next year's expected profit — higher means you pay more per dollar of earnings.",
    "Trailing P/E": "Price versus the past year's actual profit.",
    "EV/EBITDA": "Company value versus core operating profit (EBITDA); lower is generally cheaper.",
    "Price/Sales": "Price versus yearly revenue — useful when profits are thin.",
    "Revenue Growth": "How fast sales are growing year over year.",
    "Earnings Growth": "How fast profit is growing year over year.",
    "Operating Margin": "Profit left from core operations, as a share of revenue.",
    "Gross Margin": "Revenue left after the direct cost of making the product.",
    "Return on Equity": "Profit generated per dollar of shareholder money; higher is more efficient.",
    "FCF Margin (TTM)": "Free cash flow (cash left after running and investing in the business) as a share of revenue, over the last 12 months.",
    "Net Debt / EBITDA": "Debt minus cash versus yearly operating profit — roughly how many years of profit it would take to repay debt; lower is safer.",
    "Current Ratio": "Short-term assets versus short-term bills — above 1 means it can cover near-term obligations.",
    "Interest Coverage (TTM)": "How many times operating profit covered interest payments over the last 12 months; higher is safer.",
    "CAGR": "Compound annual growth rate — the average yearly return over the period.",
    "Annualized Volatility": "How much the price swings in a typical year; higher is a bumpier ride.",
    "Max Drawdown": "The worst peak-to-trough price drop over the period.",
    "Current RSI": "Relative Strength Index (0-100): above ~70 looks overbought, below ~30 oversold.",
    "Current Drawdown": "How far the price currently sits below its recent peak.",
    "Buybacks (TTM)": "Cash spent repurchasing its own shares over the last 12 months — shrinks the share count, boosting each remaining share's claim.",
    "Dividends Paid (TTM)": "Cash paid out to shareholders over the last 12 months.",
    "Capital Expenditure (TTM)": "Cash reinvested in the business (equipment, facilities, technology) over the last 12 months.",
    "Stock-Based Compensation (TTM)": "Pay issued as stock over the last 12 months — a real cost that dilutes existing shareholders.",
    "SBC % of Revenue (TTM)": "Stock pay as a share of revenue; above ~10% means meaningful ongoing dilution.",
    "Cash Conversion (OCF ÷ Net Income)": "How much of reported profit turns into actual cash; near or above 1 is healthy, well below 1 is a red flag.",
    "Free Cash Flow (TTM)": "Cash left after running and investing in the business over the last 12 months — what could fund dividends, buybacks, or debt paydown.",
}

# Short parenthetical glosses injected on first use of an abbreviation in prose
# (notes, macro flags, interpretations) so jargon always comes with a translation.
_JARGON_GLOSS = [
    ("EBITDA", "core operating profit before interest, taxes, and write-downs"),
    ("P/E", "price compared to profit"),
    ("PEG", "valuation weighed against the growth rate"),
    ("RSI", "a 0-100 momentum gauge"),
    ("CAGR", "average yearly return"),
    ("FCF", "free cash flow"),
    ("ROE", "return on equity"),
]


def humanize_jargon(text):
    """
    Add a plain-language gloss the first time each known abbreviation appears in a
    string, e.g. "...relative to EBITDA" -> "...relative to EBITDA (core operating
    profit...)". Leaves everything else untouched and won't double-gloss.
    """
    if not isinstance(text, str) or not text:
        return text
    for term, gloss in _JARGON_GLOSS:
        idx = text.find(term)
        if idx == -1:
            continue
        after = idx + len(term)
        if text[after:].lstrip().startswith("("):  # already glossed
            continue
        text = text[:after] + " (" + gloss + ")" + text[after:]
    return text


def humanize_notes(notes):
    """Apply humanize_jargon across a {category: [notes]} mapping."""
    if not isinstance(notes, dict):
        return notes
    return {k: [humanize_jargon(n) for n in (v or [])] for k, v in notes.items()}


def build_metric_snapshot(assessment):
    peer = assessment.get("peer_relative_valuation", {})
    snapshot = peer.get("target_snapshot", {}) if isinstance(peer, dict) else {}

    balance = assessment.get("balance_sheet_risk", {})
    earnings = assessment.get("earnings_quality", {})
    historical = assessment.get("historical_performance", {})
    edge = assessment.get("edge", {})
    analyst = assessment.get("analyst_expectations", {})

    price_targets = get_nested(
        analyst,
        "analyst_price_targets_from_info",
        default={}
    )

    return {
        "Company": snapshot.get("shortName"),
        "Sector": snapshot.get("sector"),
        "Industry": snapshot.get("industry"),
        "Market Cap": money(snapshot.get("marketCap")),
        "Forward P/E": num(snapshot.get("forwardPE")),
        "Trailing P/E": num(snapshot.get("trailingPE")),
        "EV/EBITDA": num(snapshot.get("enterpriseToEbitda")),
        "Price/Sales": num(snapshot.get("priceToSalesTrailing12Months")),
        "Revenue Growth": pct(snapshot.get("revenueGrowth")),
        "Earnings Growth": pct(snapshot.get("earningsGrowth")),
        "Operating Margin": pct(snapshot.get("operatingMargins")),
        "Gross Margin": pct(snapshot.get("grossMargins")),
        "FCF Margin (TTM)": pct(earnings.get("fcf_margin") if isinstance(earnings, dict) else None),
        "Net Debt / EBITDA": num(balance.get("net_debt_to_ebitda") if isinstance(balance, dict) else None),
        "Current Ratio": num(balance.get("current_ratio") if isinstance(balance, dict) else None),
        "Interest Coverage (TTM)": num(balance.get("interest_coverage") if isinstance(balance, dict) else None),
        "CAGR": pct(historical.get("cagr") if isinstance(historical, dict) else None),
        "Annualized Volatility": pct(historical.get("annualized_volatility") if isinstance(historical, dict) else None),
        "Max Drawdown": pct(historical.get("max_drawdown") if isinstance(historical, dict) else None),
        "Current RSI": num(edge.get("current_RSI") if isinstance(edge, dict) else None),
        "Current Drawdown": pct(edge.get("current_drawdown") if isinstance(edge, dict) else None),
        "Analyst Mean Target": money(price_targets.get("targetMeanPrice") if isinstance(price_targets, dict) else None),
        "Analyst Rating": price_targets.get("recommendationKey") if isinstance(price_targets, dict) else "N/A",
    }


def build_peer_comparison(assessment):
    """
    Turn the peer-relative valuation (already computed in
    retrieve_peer_relative_valuation) into a user-facing "this stock vs. peer
    median" table. This is the high-value part of the competitor data: it shows
    whether the stock trades at a premium/discount and whether its growth and
    margins lead or lag its industry peers.
    """
    peer = assessment.get("peer_relative_valuation", {})
    if not isinstance(peer, dict):
        return {"peers_used": [], "rows": []}

    snapshot = peer.get("target_snapshot") or {}
    medians = peer.get("peer_medians")
    peers_used = peer.get("peer_tickers_used") or []

    def median_of(key):
        # peer_medians is typically a pandas Series, but tolerate dict/None too.
        try:
            return clean_number(medians.get(key)) if hasattr(medians, "get") else None
        except Exception:
            return None

    # (label, snapshot key, formatter, direction) — "valuation" multiples are
    # better when lower (a premium is a caution flag); "quality" metrics are
    # better when higher.
    metrics = [
        ("Forward P/E", "forwardPE", num, "valuation"),
        ("Trailing P/E", "trailingPE", num, "valuation"),
        ("EV/EBITDA", "enterpriseToEbitda", num, "valuation"),
        ("Price/Sales", "priceToSalesTrailing12Months", num, "valuation"),
        ("Revenue Growth", "revenueGrowth", pct, "quality"),
        ("Operating Margin", "operatingMargins", pct, "quality"),
        ("Gross Margin", "grossMargins", pct, "quality"),
        ("Return on Equity", "returnOnEquity", pct, "quality"),
    ]

    rows = []
    for label, key, fmt, kind in metrics:
        value = clean_number(snapshot.get(key))
        median = median_of(key)
        if value is None or median is None or median == 0:
            continue
        # Negative multiples (e.g. P/E on a loss-maker) make ratios meaningless.
        if kind == "valuation" and (value <= 0 or median <= 0):
            continue

        rel = value / median - 1.0

        if abs(rel) < 0.10:
            vs_peers = "In line with peers"
        elif kind == "valuation":
            vs_peers = (
                f"Premium (+{rel:.0%} vs peers)" if rel > 0
                else f"Discount ({rel:.0%} vs peers)"
            )
        else:  # quality
            vs_peers = (
                f"Above peers (+{rel:.0%})" if rel > 0
                else f"Below peers ({rel:.0%})"
            )

        rows.append({
            "metric": label,
            "value": fmt(value),
            "peer_median": fmt(median),
            "vs_peers": vs_peers,
        })

    return {"peers_used": peers_used, "rows": rows}


# Sectors most exposed to a recessionary / inverted-curve signal (yfinance labels).
_CYCLICAL_SECTORS = {
    "Consumer Cyclical", "Financial Services", "Basic Materials",
    "Industrials", "Energy", "Real Estate",
}


def score_macro_overlay(snapshot, balance, regime):
    """
    Macro risk is identical for every stock on a given day, so a flat macro penalty
    would add no power to *discriminate between stocks*. Instead, each macro
    condition only bites when it interacts with a vulnerability the specific stock
    actually has (high beta, high leverage, or a rich multiple). This is what turns
    macro data from decoration into a real, stock-specific risk signal.

    Returns (risk_adjustment, notes); risk_adjustment is added to the risk score.
    """
    if not isinstance(regime, dict) or "error" in regime:
        return 0.0, []

    flags = regime.get("flags", {}) or {}
    snapshot = snapshot if isinstance(snapshot, dict) else {}
    balance = balance if isinstance(balance, dict) else {}

    beta = clean_number(snapshot.get("beta"))
    forward_pe = clean_number(snapshot.get("forwardPE"))
    sector = snapshot.get("sector")
    net_debt_ebitda = clean_number(balance.get("net_debt_to_ebitda"))

    adjustment = 0.0
    notes = []

    # 1) Inverted curve (recession signal) bites cyclical / high-beta names.
    if flags.get("curve_inverted"):
        if beta is not None and beta > 1.2:
            adjustment += 1.0
            notes.append(
                f"Yield curve is inverted (a recession signal) and this is a high-beta "
                f"name (beta {beta:.2f}); cyclical downside risk is elevated."
            )
        elif sector in _CYCLICAL_SECTORS:
            adjustment += 0.5
            notes.append(
                f"Yield curve is inverted (a recession signal) and {sector} is a "
                f"cyclical sector; earnings are more exposed to a slowdown."
            )

    # 2) Wide credit spreads bite highly-levered balance sheets (refinancing risk).
    if flags.get("credit_stress") and net_debt_ebitda is not None and net_debt_ebitda > 3.0:
        adjustment += 1.0
        notes.append(
            f"High-yield credit spreads are wide and leverage is high "
            f"(net debt/EBITDA {net_debt_ebitda:.1f}x); refinancing risk is elevated."
        )

    # 3) Elevated policy rates bite long-duration, expensive growth (high forward P/E).
    if flags.get("rates_elevated") and forward_pe is not None and forward_pe > 30:
        adjustment += 1.0
        notes.append(
            f"Policy rates are elevated and the stock is richly valued "
            f"(forward P/E {forward_pe:.0f}); high-multiple growth is rate-sensitive."
        )

    # Cap so the macro overlay informs the score without dominating it.
    adjustment = min(adjustment, 2.5)

    if not notes:
        notes.append("No stock-specific macro risks flagged under the current regime.")

    return adjustment, notes


def generate_scored_summary(assessment):
    ticker = assessment.get("ticker", "UNKNOWN")

    peer = assessment.get("peer_relative_valuation", {})
    snapshot = peer.get("target_snapshot", {}) if isinstance(peer, dict) else {}
    peer_medians = peer.get("peer_medians") if isinstance(peer, dict) else {}

    balance_score, balance_notes = score_balance_sheet(
        assessment.get("balance_sheet_risk", {})
    )

    earnings_score, earnings_notes = score_earnings_quality(
        assessment.get("earnings_quality", {})
    )

    valuation_score, valuation_notes = score_valuation(
        snapshot,
        peer_medians
    )

    growth_score, growth_notes = score_growth_quality(snapshot)

    market_score, market_notes = score_market_technical(
        assessment.get("edge", {}),
        assessment.get("historical_performance", {}),
        assessment.get("short_interest", {}),
    )

    governance_score, governance_notes = score_governance(
        assessment.get("governance", {})
    )

    category_scores = {
        "Valuation": valuation_score,
        "Growth / Profitability": growth_score,
        "Balance Sheet": balance_score,
        "Earnings Quality": earnings_score,
        "Market / Technical": market_score,
        "Governance": governance_score,
    }

    available_scores = {
        k: v for k, v in category_scores.items()
        if v is not None
    }

    if available_scores:
        overall_quality_score = sum(available_scores.values()) / len(available_scores)
    else:
        overall_quality_score = 5

    # Risk score is inverted from quality, then adjusted for extreme conditions.
    generalized_risk_score = 10 - overall_quality_score

    historical = assessment.get("historical_performance", {})
    if isinstance(historical, dict):
        vol = historical.get("annualized_volatility")
        drawdown = historical.get("max_drawdown")

        if vol is not None and vol > 0.50:
            generalized_risk_score += 1
        if drawdown is not None and drawdown < -0.60:
            generalized_risk_score += 1

    earnings = assessment.get("earnings_quality", {})
    if isinstance(earnings, dict):
        if earnings.get("free_cash_flow_latest") is not None and earnings.get("free_cash_flow_latest") < 0:
            generalized_risk_score += 1

    balance = assessment.get("balance_sheet_risk", {})
    if isinstance(balance, dict):
        if balance.get("net_debt_to_ebitda") is not None and balance.get("net_debt_to_ebitda") > 4:
            generalized_risk_score += 1

    # Macro overlay: only adds risk where the current regime interacts with this
    # stock's specific vulnerabilities (high beta, high leverage, rich multiple).
    macro_regime = assessment.get("macro_regime", {})
    macro_adjustment, macro_notes = score_macro_overlay(snapshot, balance, macro_regime)
    generalized_risk_score += macro_adjustment

    generalized_risk_score = round(clamp(generalized_risk_score), 1)
    overall_quality_score = round(clamp(overall_quality_score), 1)

    if generalized_risk_score <= 3:
        risk_label = "Low"
    elif generalized_risk_score <= 6:
        risk_label = "Moderate"
    elif generalized_risk_score <= 8:
        risk_label = "High"
    else:
        risk_label = "Very High"

    if overall_quality_score >= 7.5:
        verdict = "Attractive / strong profile"
    elif overall_quality_score >= 6:
        verdict = "Generally solid, but not risk-free"
    elif overall_quality_score >= 4.5:
        verdict = "Mixed / needs deeper research"
    else:
        verdict = "Weak or speculative profile"

    return {
        "ticker": ticker,
        "verdict": verdict,
        "overall_quality_score": overall_quality_score,
        "generalized_risk_score": generalized_risk_score,
        "risk_label": risk_label,
        "category_scores": category_scores,
        "metric_snapshot": build_metric_snapshot(assessment),
        "notes": {
            "Valuation": valuation_notes,
            "Growth / Profitability": growth_notes,
            "Balance Sheet": balance_notes,
            "Earnings Quality": earnings_notes,
            "Market / Technical": market_notes,
            "Governance": governance_notes,
        },
        "macro_context": {
            "regime": macro_regime.get("label") if isinstance(macro_regime, dict) else None,
            "stock_specific_notes": macro_notes,
            "risk_adjustment": round(macro_adjustment, 1),
        },
    }


def print_concise_scored_summary(assessment):
    summary = generate_scored_summary(assessment)

    print("\n" + "=" * 80)
    print(f"{summary['ticker']} STOCK ASSESSMENT")
    print("=" * 80)

    print(f"\nVerdict: {summary['verdict']}")
    print(f"Overall Quality Score: {summary['overall_quality_score']}/10")
    print(
        f"Generalized Risk Score: {summary['generalized_risk_score']}/10 "
        f"({summary['risk_label']} Risk)"
    )

    print("\n" + "-" * 80)
    print("KEY METRICS")
    print("-" * 80)

    metrics = summary["metric_snapshot"]

    for key, value in metrics.items():
        if value is not None:
            print(f"{key}: {value}")

    print("\n" + "-" * 80)
    print("CATEGORY SCORES")
    print("-" * 80)

    for category, score in summary["category_scores"].items():
        score_text = "N/A" if score is None else f"{score}/10"
        print(f"{category}: {score_text}")

    print("\n" + "-" * 80)
    print("SUMMARY NOTES")
    print("-" * 80)

    for category, notes in summary["notes"].items():
        print(f"\n{category}:")
        for note in notes[:4]:
            print(f"  - {note}")

    print("\n" + "-" * 80)
    print("INTERPRETATION")
    print("-" * 80)

    print(
        "This is a generalized rule-based screen, not a buy/sell recommendation. "
        "Use it to identify where deeper research is needed: valuation, balance sheet, "
        "earnings quality, competitive position, or market risk."
    )

def score_label(score):
    if score is None:
        return "unavailable"

    try:
        score = float(score)
    except Exception:
        return "unavailable"

    if score >= 7.5:
        return "strong"
    if score >= 6:
        return "solid"
    if score >= 4.5:
        return "mixed"
    return "weak"


def build_key_takeaway(summary):
    ticker = summary.get("ticker", "This stock")
    verdict = summary.get("verdict", "Mixed / needs deeper research")
    quality = summary.get("overall_quality_score", "N/A")
    risk = summary.get("generalized_risk_score", "N/A")
    risk_label = summary.get("risk_label", "Unknown")

    scores = summary.get("category_scores", {})

    strongest = None
    weakest = None

    numeric_scores = {
        k: v for k, v in scores.items()
        if isinstance(v, (int, float))
    }

    if numeric_scores:
        strongest = max(numeric_scores, key=numeric_scores.get)
        weakest = min(numeric_scores, key=numeric_scores.get)

    if strongest and weakest:
        return (
            f"{ticker} screens as {verdict.lower()}, with an overall quality score of "
            f"{quality}/10 and {risk_label.lower()} risk ({risk}/10); its strongest area is "
            f"{strongest.lower()}, while {weakest.lower()} factors deserve the most follow-up research."
        )

    return (
        f"{ticker} screens as {verdict.lower()}, with an overall quality score of "
        f"{quality}/10 and {risk_label.lower()} risk ({risk}/10), based on the available data."
    )


def build_category_interpretations(summary):
    scores = summary.get("category_scores", {})

    return {
        "Valuation": (
            f"{score_label(scores.get('Valuation')).capitalize()} valuation score. "
            "This compares whether the stock looks cheap, fair, or expensive by weighing its price against its profits, sales, and cash flow, and against what similar companies trade for."
        ),
        "Growth / Profitability": (
            f"{score_label(scores.get('Growth / Profitability')).capitalize()} growth and profitability score. "
            "This shows whether the company is growing revenue and earnings while maintaining attractive margins and returns."
        ),
        "Balance Sheet": (
            f"{score_label(scores.get('Balance Sheet')).capitalize()} balance-sheet score. "
            "This evaluates debt, cash, liquidity, leverage, and interest coverage to estimate financial stability."
        ),
        "Earnings Quality": (
            f"{score_label(scores.get('Earnings Quality')).capitalize()} earnings-quality score. "
            "This checks whether reported profits are supported by operating cash flow and free cash flow."
        ),
        "Market / Technical": (
            f"{score_label(scores.get('Market / Technical')).capitalize()} market and technical score. "
            "This looks at trend, momentum, volatility, drawdown, and short interest to understand current market behavior."
        ),
        "Governance": (
            f"{score_label(scores.get('Governance')).capitalize()} governance score. "
            "This uses available governance-risk fields to flag possible concerns around board, audit, compensation, or shareholder-rights risk."
        ),
    }


def build_risks_and_strengths(summary):
    """
    Split the assessment into a short, scannable list of the biggest strengths and
    the biggest risks, each with a one-line reason. Drives the side-by-side
    "Key Risks & Strengths" panel.
    """
    scores = summary.get("category_scores", {})
    notes = summary.get("notes", {})

    def first_note(category):
        items = notes.get(category) or []
        for n in items:
            if isinstance(n, str) and n.strip() and "unavailable" not in n.lower():
                return n
        return None

    rated = [(c, s) for c, s in scores.items() if isinstance(s, (int, float))]
    rated.sort(key=lambda x: x[1], reverse=True)

    strengths = []
    for category, score in rated:
        if score >= 6.5 and len(strengths) < 3:
            strengths.append({
                "title": category,
                "detail": first_note(category) or f"Scores {score}/10 in this category.",
            })

    risks = []
    for category, score in sorted(rated, key=lambda x: x[1]):
        if score <= 5.5 and len(risks) < 3:
            risks.append({
                "title": category,
                "detail": first_note(category) or f"Scores only {score}/10 in this category.",
            })

    # Fold in any stock-specific macro risks the overlay flagged.
    macro = summary.get("macro_context", {}) or {}
    for note in macro.get("stock_specific_notes", []) or []:
        if isinstance(note, str) and "no stock-specific macro" not in note.lower() and len(risks) < 4:
            risks.append({"title": "Macro Exposure", "detail": note})

    # Guarantee each side has at least one entry so the panel never looks broken.
    if not strengths and rated:
        c, s = rated[0]
        strengths.append({"title": c, "detail": first_note(c) or f"Relatively strongest area ({s}/10)."})
    if not risks and rated:
        c, s = min(rated, key=lambda x: x[1])
        risks.append({"title": c, "detail": first_note(c) or f"Relatively weakest area ({s}/10)."})

    return {"strengths": strengths, "risks": risks}


def build_investor_fit(summary):
    """
    Plain-language "what this means for you", as 1-2 short bullet points: which
    kind of investor the stock's risk/reward profile tends to suit, and who should
    be cautious. Returns a list of strings.
    """
    scores = summary.get("category_scores", {})
    ticker = summary.get("ticker", "This stock")
    risk = summary.get("generalized_risk_score")
    risk_label = (summary.get("risk_label") or "moderate").lower()

    growth = scores.get("Growth / Profitability")
    valuation = scores.get("Valuation")
    balance = scores.get("Balance Sheet")

    def strong(v, threshold=6.5):
        return isinstance(v, (int, float)) and v >= threshold

    audiences = []
    if strong(growth):
        audiences.append("growth- and profitability-focused investors")
    if strong(valuation):
        audiences.append("value-conscious investors looking for a reasonable price")
    if strong(balance, 7) and isinstance(risk, (int, float)) and risk <= 4:
        audiences.append("conservative investors who prioritize stability")

    if not audiences:
        audiences.append("investors comfortable with a mixed risk/reward profile")

    if len(audiences) == 1:
        audience_text = audiences[0]
    else:
        audience_text = ", ".join(audiences[:-1]) + " and " + audiences[-1]

    if isinstance(risk, (int, float)) and risk >= 6.5:
        caveat = ("Its higher risk makes it a weaker fit for cautious investors who "
                  "are uncomfortable with big price swings or drops.")
    elif isinstance(risk, (int, float)) and risk >= 4:
        caveat = ("With moderate risk, more cautious investors should keep positions "
                  "small and keep an eye on the risks flagged below.")
    else:
        caveat = "Its lower risk profile is relatively friendly to cautious investors."

    return [
        f"With {risk_label} risk ({risk}/10), {ticker} may appeal most to {audience_text}.",
        caveat,
    ]

def build_capital_allocation_view(assessment) -> List[Dict[str, str]]:
    """Format the capital-allocation source as display rows for the web UI.
    Cash outflows (buybacks, dividends, capex) are reported negative in the
    statement; show them as positive spend amounts."""
    cap = assessment.get("capital_allocation")
    if not isinstance(cap, dict) or "error" in cap:
        return []

    def outflow(value):
        return money(abs(value)) if value is not None else None

    rows = [
        ("Buybacks (TTM)", outflow(cap.get("buybacks_latest"))),
        ("Dividends Paid (TTM)", outflow(cap.get("dividends_paid_latest"))),
        ("Capital Expenditure (TTM)", outflow(cap.get("capital_expenditure_latest"))),
        ("Stock-Based Compensation (TTM)", outflow(cap.get("stock_based_compensation_latest"))),
        ("SBC % of Revenue (TTM)", pct(cap.get("sbc_as_percent_of_revenue"))),
        ("Cash Conversion (OCF ÷ Net Income)", num(cap.get("cash_conversion_ocf_to_net_income"))),
        ("Free Cash Flow (TTM)", money(cap.get("free_cash_flow_from_info"))),
    ]
    return [
        {"label": label, "value": value}
        for label, value in rows
        if value not in (None, "N/A")
    ]


def build_industry_kpi_view(assessment) -> Optional[Dict[str, Any]]:
    """Industry-specific KPIs worth researching, for the 'what to watch' panel.
    Returns None when we have no suggestions for this industry."""
    kpi = assessment.get("industry_kpi_framework")
    if not isinstance(kpi, dict) or "error" in kpi:
        return None
    suggested = kpi.get("suggested_kpis_to_research") or []
    if not suggested:
        return None
    return {
        "industry": kpi.get("industry") or kpi.get("sector"),
        "kpis": suggested,
        "note": (
            "Our score can't see these — they come from filings, earnings decks, "
            "and investor presentations. If you research one thing beyond this "
            "page, make it these."
        ),
    }


def get_stock_assessment_for_html(ticker: str) -> Dict[str, Any]:
    """
    Runs the full stock assessment and returns the same information that
    print_concise_scored_summary() prints, but as a dictionary for HTML/API use.

    macro/FRED data is skipped here: it is the single most expensive source
    (~30-70s, dominated by two daily FRED treasury series that read-time-out) and
    nothing in the summary, metric snapshot, scores, or interpretations actually
    consumes it. Skipping it cuts the web request time by more than half with no
    change to the displayed result. (The CLI still fetches it via include_macro.)
    """
    # Reject nonexistent tickers before the expensive pipeline runs. Without
    # this, a typo burned ~18s of doomed Yahoo calls, returned a fabricated
    # "success" score built from zero data, and — because the dead .info
    # attempts look identical to throttling — tripped the circuit breaker,
    # degrading real requests for minutes. The probe uses the chart endpoint
    # (cheap, and on a separate rate-limit budget from .info), so a fake
    # ticker never touches the breaker.
    ticker = ticker.upper().strip()
    probe = safe_call(lambda: get_ticker(ticker).history(period="5d"), label="existence_probe")
    if not isinstance(probe, pd.DataFrame) or probe.empty:
        return {
            "success": False,
            "ticker": ticker,
            "error": (
                f"We couldn't find a stock with the ticker \"{ticker}\". "
                "Double-check the symbol, or start typing in the search box "
                "and pick from the suggestions."
            ),
        }

    # lean=True + the include flags skip every source the web output never
    # reads (news, technical, broad market, SEC, options, ownership, ...) —
    # a 10-20s saving per request with an identical displayed result.
    assessment = comprehensive_stock_assessment(
        ticker,
        include_macro=False,
        include_sec=False,
        include_options=False,
        lean=True,
    )
    summary = generate_scored_summary(assessment)

    # Translate jargon into plain language before anything reads the notes, so the
    # strengths/risks bullets and the category breakdown both inherit the glosses.
    summary["notes"] = humanize_notes(summary.get("notes", {}))
    macro_ctx = summary.get("macro_context") or {}
    macro_ctx["stock_specific_notes"] = [
        humanize_jargon(n) for n in (macro_ctx.get("stock_specific_notes") or [])
    ]

    key_takeaway = build_key_takeaway(summary)
    category_interpretations = build_category_interpretations(summary)
    peer_comparison = build_peer_comparison(assessment)
    risks_and_strengths = build_risks_and_strengths(summary)
    investor_fit = build_investor_fit(summary)

    # Company display name for the UI (e.g. saved-analysis cards). The Ticker is
    # already cached from the assessment run, so .info costs no extra network call.
    info = get_cached_info(summary["ticker"])
    company_name = info.get("shortName") or info.get("longName") or ""

    # Data-sufficiency gate. Price history alone can exist for shell companies,
    # SPAC remnants, and thinly traded OTC listings (and .info can vanish during
    # a Yahoo throttle). A risk score computed with zero fundamentals is
    # fabrication, not analysis — refuse honestly instead of rendering it.
    snapshot = summary["metric_snapshot"]
    fundamentals_present = sum(
        1
        for key in (
            "Market Cap", "Trailing P/E", "Forward P/E", "Operating Margin",
            "Gross Margin", "Revenue Growth", "Sector",
        )
        if snapshot.get(key) not in (None, "N/A")
    )
    if not company_name and fundamentals_present == 0:
        return {
            "success": False,
            "ticker": summary["ticker"],
            "error": (
                f"\"{summary['ticker']}\" trades, but we couldn't find enough company "
                "fundamentals to build a meaningful analysis — it may be a shell "
                "company or a very thinly traded listing. If you believe this is a "
                "real company, try again in a few minutes; our data source may be "
                "temporarily limited."
            ),
        }

    # Delayed quote for display. Comes from the same cached .info the analysis
    # already fetched (zero extra network cost). Yahoo's quote data is
    # exchange-delayed and our cache adds up to an hour — the UI labels it as
    # delayed with a timestamp; deliberately NOT a live-updating price.
    current_price = clean_number(
        info.get("currentPrice") or info.get("regularMarketPrice")
    )
    previous_close = clean_number(
        info.get("regularMarketPreviousClose") or info.get("previousClose")
    )
    quote = None
    if current_price is not None:
        quote = {
            "price": current_price,
            "change_pct": (
                (current_price - previous_close) / previous_close
                if previous_close
                else None
            ),
            "currency": info.get("currency") or "USD",
            "time": info.get("regularMarketTime"),
        }

    return make_json_safe({
        "success": True,
        "ticker": summary["ticker"],
        "company_name": company_name,
        "quote": quote,
        "verdict": summary["verdict"],
        "overall_quality_score": summary["overall_quality_score"],
        "generalized_risk_score": summary["generalized_risk_score"],
        "risk_label": summary["risk_label"],
        "key_takeaway": key_takeaway,
        "investor_fit": investor_fit,
        "strengths": risks_and_strengths["strengths"],
        "risks": risks_and_strengths["risks"],
        "metric_snapshot": summary["metric_snapshot"],
        "metric_glossary": METRIC_GLOSSARY,
        "category_scores": summary["category_scores"],
        "summary_notes": summary["notes"],
        "category_interpretations": category_interpretations,
        "peer_comparison": peer_comparison,
        "macro_context": macro_ctx,
        "capital_allocation": build_capital_allocation_view(assessment),
        "industry_kpis": build_industry_kpi_view(assessment),
        "interpretation": (
            "This is a generalized rule-based screen, not a buy/sell recommendation. "
            "Use it to identify where deeper research is needed: valuation, balance sheet, "
            "earnings quality, competitive position, or market risk."
        )
    })

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run stock assessment.")
    parser.add_argument("ticker", nargs="?", help="Stock ticker, e.g. AAPL")
    parser.add_argument("--json", action="store_true", help="Print JSON instead of console report")
    parser.add_argument("--output", help="Optional path to save JSON output, e.g. output.json")
    parser.add_argument("--full", action="store_true", help="Include full raw assessment in JSON output")

    args = parser.parse_args()

    user_ticker = args.ticker or input("Enter ticker: ").upper().strip()

    if args.json or args.output:
        result = get_stock_assessment_for_html(
            ticker=user_ticker,
            include_full_assessment=args.full,
        )

        json_text = json.dumps(result, indent=2)

        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(json_text)

        if args.json:
            print(json_text)

    else:
        result = comprehensive_stock_assessment(user_ticker)
        print_concise_scored_summary(result)