from __future__ import annotations

import json
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
    stock = yf.Ticker(ticker)

    return {
        "price_history": stock.history(period="5y"),
        "info": stock.info,
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
    stock = yf.Ticker(ticker)
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

    output = {}

    for name, fred_code in fred_series.items():
        output[name] = safe_call(
            lambda code=fred_code: pdr.DataReader(code, "fred", start),
            label=f"FRED:{fred_code}",
        )

    return output


# ============================================================
# 5. Industry and competitor data
# ============================================================

def retrieve_industry_competitor_data(
    ticker: str,
    max_competitors: int = 30
) -> Dict[str, Any]:
    info = yf.Ticker(ticker).info

    industry = info.get("industry")
    sector = info.get("sector")

    eq = Equities()
    df = eq.select()

    competitors = df[
        (df.get("industry") == industry) |
        (df.get("sector") == sector)
    ].copy()

    return {
        "sector": sector,
        "industry": industry,
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
    stock = yf.Ticker(ticker)

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
    stock = yf.Ticker(ticker)

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
            k: stock.info.get(k)
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
    info = yf.Ticker(ticker).info

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

    ticker_col = None

    for candidate in ["symbol", "ticker", "Symbol", "Ticker"]:
        if isinstance(competitors, pd.DataFrame) and candidate in competitors.columns:
            ticker_col = candidate
            break

    peer_tickers = []

    if ticker_col:
        peer_tickers = [
            str(x).upper()
            for x in competitors[ticker_col].dropna().unique()
        ]

        peer_tickers = [
            x for x in peer_tickers
            if x != ticker.upper()
        ][:max_peers]

    snapshots = {
        ticker.upper(): get_company_snapshot(ticker)
    }

    for peer in peer_tickers:
        snapshots[peer] = safe_call(
            lambda p=peer: get_company_snapshot(p),
            label=f"peer_snapshot:{peer}",
        )

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
    stock = yf.Ticker(ticker)

    cashflow = stock.cashflow
    income = stock.income_stmt
    balance = stock.balance_sheet
    info = stock.info

    buybacks = latest_value(cashflow, "Repurchase Of Capital Stock")
    dividends_paid = latest_value(cashflow, "Cash Dividends Paid")
    capex = latest_value(cashflow, "Capital Expenditure")
    free_cash_flow = info.get("freeCashflow")
    operating_cash_flow = latest_value(cashflow, "Operating Cash Flow")
    net_income = latest_value(income, "Net Income")
    stock_based_comp = latest_value(cashflow, "Stock Based Compensation")
    revenue = latest_value(income, "Total Revenue")

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
        "retained_earnings_latest": latest_value(balance, "Retained Earnings"),
    }


def analyze_earnings_quality(ticker: str) -> Dict[str, Any]:
    stock = yf.Ticker(ticker)

    cashflow = stock.cashflow
    income = stock.income_stmt
    balance = stock.balance_sheet

    revenue = latest_value(income, "Total Revenue")
    net_income = latest_value(income, "Net Income")
    ocf = latest_value(cashflow, "Operating Cash Flow")
    fcf = latest_value(cashflow, "Free Cash Flow")
    receivables = latest_value(balance, "Accounts Receivable")
    inventory = latest_value(balance, "Inventory")

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
    stock = yf.Ticker(ticker)

    info = stock.info
    balance = stock.balance_sheet
    income = stock.income_stmt

    total_debt = info.get("totalDebt") or latest_value(balance, "Total Debt")
    cash = info.get("totalCash") or latest_value(balance, "Cash And Cash Equivalents")
    ebitda = info.get("ebitda")

    operating_income = latest_value(income, "Operating Income")
    interest_expense = latest_value(income, "Interest Expense")

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
    stock = yf.Ticker(ticker)

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
    stock = yf.Ticker(ticker)

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
    info = yf.Ticker(ticker).info

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
    info = yf.Ticker(ticker).info

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
    stock = yf.Ticker(ticker)
    info = stock.info

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
    info = yf.Ticker(ticker).info
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
    info = yf.Ticker(ticker).info
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
) -> Dict[str, Any]:
    ticker = ticker.upper().strip()

    assessment = {
        "ticker": ticker,
        "quantitative": safe_call(
            lambda: retrieve_quantitative_data(ticker),
            label="quantitative",
        ),
        "news_and_sentiment": safe_call(
            lambda: retrieve_basic_news_sentiment(ticker),
            label="news_and_sentiment",
        ),
        "technical": safe_call(
            lambda: retrieve_technical_indicators(ticker),
            label="technical",
        ),
        "broad_market": safe_call(
            retrieve_broad_market_data,
            label="broad_market",
        ),
        "industry_competitors": safe_call(
            lambda: retrieve_industry_competitor_data(ticker),
            label="industry_competitors",
        ),
        "risk_factors": safe_call(
            lambda: retrieve_risk_factors(ticker),
            label="risk_factors",
        ),
        "analyst_expectations": safe_call(
            lambda: retrieve_analyst_expectations(ticker),
            label="analyst_expectations",
        ),
        "peer_relative_valuation": safe_call(
            lambda: retrieve_peer_relative_valuation(ticker),
            label="peer_relative_valuation",
        ),
        "capital_allocation": safe_call(
            lambda: analyze_capital_allocation(ticker),
            label="capital_allocation",
        ),
        "earnings_quality": safe_call(
            lambda: analyze_earnings_quality(ticker),
            label="earnings_quality",
        ),
        "balance_sheet_risk": safe_call(
            lambda: analyze_balance_sheet_risk(ticker),
            label="balance_sheet_risk",
        ),
        "ownership_and_insiders": safe_call(
            lambda: retrieve_ownership_and_insider_data(ticker),
            label="ownership_and_insiders",
        ),
        "short_interest": safe_call(
            lambda: retrieve_short_interest_data(ticker),
            label="short_interest",
        ),
        "industry_kpi_framework": safe_call(
            lambda: identify_industry_kpis(ticker),
            label="industry_kpi_framework",
        ),
        "governance": safe_call(
            lambda: retrieve_governance_data(ticker),
            label="governance",
        ),
        "geographic_regulatory_exposure": safe_call(
            lambda: retrieve_exposure_checklist(ticker),
            label="geographic_regulatory_exposure",
        ),
        "alternative_data_checklist": alternative_data_checklist(ticker),
        "historical_performance": safe_call(
            lambda: analyze_historical_performance(ticker),
            label="historical_performance",
        ),
        "portfolio_context": safe_call(
            lambda: analyze_portfolio_context(ticker, current_holdings),
            label="portfolio_context",
        ),
        "edge": safe_call(
            lambda: assess_stock_edge(ticker),
            label="edge",
        ),
        "niche": safe_call(
            lambda: check_stock_niche(ticker),
            label="niche",
        ),
    }

    if include_macro:
        assessment["macro"] = safe_call(
            retrieve_macro_data,
            label="macro",
        )

    if include_sec:
        assessment["sec_filings"] = safe_call(
            lambda: retrieve_sec_filings(ticker),
            label="sec_filings",
        )

    if include_options:
        assessment["options_market"] = safe_call(
            lambda: retrieve_options_market_data(ticker),
            label="options_market",
        )

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
        "FCF Margin": pct(earnings.get("fcf_margin") if isinstance(earnings, dict) else None),
        "Net Debt / EBITDA": num(balance.get("net_debt_to_ebitda") if isinstance(balance, dict) else None),
        "Current Ratio": num(balance.get("current_ratio") if isinstance(balance, dict) else None),
        "Interest Coverage": num(balance.get("interest_coverage") if isinstance(balance, dict) else None),
        "CAGR": pct(historical.get("cagr") if isinstance(historical, dict) else None),
        "Annualized Volatility": pct(historical.get("annualized_volatility") if isinstance(historical, dict) else None),
        "Max Drawdown": pct(historical.get("max_drawdown") if isinstance(historical, dict) else None),
        "Current RSI": num(edge.get("current_RSI") if isinstance(edge, dict) else None),
        "Current Drawdown": pct(edge.get("current_drawdown") if isinstance(edge, dict) else None),
        "Analyst Mean Target": money(price_targets.get("targetMeanPrice") if isinstance(price_targets, dict) else None),
        "Analyst Rating": price_targets.get("recommendationKey") if isinstance(price_targets, dict) else "N/A",
    }


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

def get_stock_assessment_for_html(ticker: str) -> Dict[str, Any]:
    """
    Runs the full stock assessment and returns the same information that
    print_concise_scored_summary() prints, but as a dictionary for HTML/API use.
    """
    assessment = comprehensive_stock_assessment(ticker)
    summary = generate_scored_summary(assessment)

    return make_json_safe({
        "success": True,
        "ticker": summary["ticker"],
        "verdict": summary["verdict"],
        "overall_quality_score": summary["overall_quality_score"],
        "generalized_risk_score": summary["generalized_risk_score"],
        "risk_label": summary["risk_label"],
        "metric_snapshot": summary["metric_snapshot"],
        "category_scores": summary["category_scores"],
        "summary_notes": summary["notes"],
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
