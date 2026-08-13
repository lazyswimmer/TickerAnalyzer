"""
SEC EDGAR + Stooq fallback data source.

Why this exists: the deployed server shares its outbound IP with other cloud
tenants, and Yahoo rate-limits quoteSummary (.info) per IP — so every analysis
died with "data source temporarily limited" even though nothing was wrong with
the code. SEC EDGAR and Stooq don't block datacenter IPs (SEC asks only for a
descriptive User-Agent), so when Yahoo won't talk to us we rebuild the
essential .info fields from the company's own filings:

  - identity:     EDGAR `submissions` API (company name, SIC industry)
  - fundamentals: EDGAR `companyfacts` XBRL API — latest fiscal year, plus the
                  prior year for growth rates
  - price:        the caller passes the last close (Yahoo's chart endpoint is
                  on a separate rate-limit budget and usually still works);
                  `stooq_history` is the backstop for price series

The output is a dict shaped like yfinance's .info — the same keys the scoring
layer already reads — tagged with `_source: "sec_edgar"` so the UI can say
where the numbers came from. Values are as-of the last annual report, so
they're staler than Yahoo's TTM figures: right for a fallback, not the
primary path. Anything we can't derive is simply absent, which the scoring
already treats as missing data.
"""

from __future__ import annotations

import io
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests

EDGAR_TIMEOUT = 20

# Flow entries (revenue, income, cash flow) qualify as "annual" when their
# start->end duration is roughly a fiscal year.
_ANNUAL_MIN_DAYS = 330
_ANNUAL_MAX_DAYS = 400


def _get_json(url: str, user_agent: str) -> Dict[str, Any]:
    res = requests.get(url, headers={"User-Agent": user_agent}, timeout=EDGAR_TIMEOUT)
    res.raise_for_status()
    return res.json()


def _parse_date(s: Optional[str]) -> Optional[datetime]:
    try:
        return datetime.strptime(s, "%Y-%m-%d")
    except Exception:
        return None


def _tag_entry_lists(facts: Dict[str, Any], tags: List[str]) -> List[List[Dict[str, Any]]]:
    """Entry lists for EVERY candidate tag that exists (us-gaap then ifrs-full,
    USD units ONLY). All candidates matter: companies switch XBRL tags
    over the years, and a tag can hold nothing but stale history — NVDA's
    revenue lived under a tag that went dead in FY2022, so 'first tag that
    exists' quietly served four-year-old numbers.

    USD-only is deliberate, not a preference: a 20-F filer reporting in euros
    would otherwise get EUR fundamentals divided into a USD market cap — a
    silently WRONG ratio, which is worse than a missing one. Foreign filers
    without USD facts simply get no fallback."""
    lists = []
    for taxonomy in ("us-gaap", "ifrs-full"):
        tax = facts.get(taxonomy) or {}
        for tag in tags:
            entries = ((tax.get(tag) or {}).get("units") or {}).get("USD")
            if entries:
                lists.append(entries)
    return lists


def _annual_by_end(entries: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """period-end -> most-recently-filed annual (~1y duration) entry."""
    annual: Dict[str, Dict[str, Any]] = {}
    for e in entries:
        start, end = _parse_date(e.get("start")), _parse_date(e.get("end"))
        if not start or not end or e.get("val") is None:
            continue
        if not (_ANNUAL_MIN_DAYS <= (end - start).days <= _ANNUAL_MAX_DAYS):
            continue
        key = e["end"]
        if key not in annual or (e.get("filed") or "") > (annual[key].get("filed") or ""):
            annual[key] = e
    return annual


def _latest_fiscal_year_end(facts: Dict[str, Any], tags: List[str]) -> Optional[str]:
    """Most recent annual period end filed under any candidate tag."""
    ends: List[str] = []
    for entries in _tag_entry_lists(facts, tags):
        ends.extend(_annual_by_end(entries))
    return max(ends) if ends else None


def _annual_at(
    facts: Dict[str, Any],
    tags: List[str],
    ref_end: Optional[str],
    offset_days: int = 0,
    tolerance_days: int = 65,
) -> Optional[float]:
    """Annual value whose period ends near ref_end minus offset_days (offset
    365 = the prior fiscal year, for growth rates). Anchoring every flow
    metric to the same reference period keeps ratios from silently mixing
    fiscal years when a company's tags cover different date ranges."""
    if ref_end is None:
        return None
    target = _parse_date(ref_end) - timedelta(days=offset_days)
    best: Optional[Tuple[int, str, float]] = None  # (gap, filed, val)
    for entries in _tag_entry_lists(facts, tags):
        for end, e in _annual_by_end(entries).items():
            gap = abs((_parse_date(end) - target).days)
            if gap > tolerance_days:
                continue
            filed = e.get("filed") or ""
            if best is None or gap < best[0] or (gap == best[0] and filed > best[1]):
                best = (gap, filed, float(e["val"]))
    return best[2] if best else None


# Discrete quarters (10-Q flow facts) run ~90 days.
_QUARTER_MIN_DAYS = 75
_QUARTER_MAX_DAYS = 105


def _quarters_by_end(facts: Dict[str, Any], tags: List[str]) -> Dict[str, Dict[str, Any]]:
    """period-end -> most-recently-filed discrete-quarter entry, merged across
    all candidate tags (same dedupe rule as _annual_by_end)."""
    quarters: Dict[str, Dict[str, Any]] = {}
    for entries in _tag_entry_lists(facts, tags):
        for e in entries:
            start, end = _parse_date(e.get("start")), _parse_date(e.get("end"))
            if not start or not end or e.get("val") is None:
                continue
            if not (_QUARTER_MIN_DAYS <= (end - start).days <= _QUARTER_MAX_DAYS):
                continue
            key = e["end"]
            if key not in quarters or (e.get("filed") or "") > (quarters[key].get("filed") or ""):
                quarters[key] = e
    return quarters


def _ttm_at(facts: Dict[str, Any], tags: List[str], fy_end: Optional[str]) -> Tuple[Optional[float], Optional[str]]:
    """(trailing-twelve-month value, as-of period end) for a flow concept.

    Companies file discrete quarters only for Q1-Q3 (the 10-K subsumes Q4), so
    naive last-4-quarters summing rarely works. The standard construction:

        TTM = anchored FY + quarters reported AFTER the FY end
                          - the matching quarters one year earlier

    Falls back to the plain FY value whenever a stub quarter lacks its
    prior-year counterpart: stale-but-correct beats fresh-but-wrong."""
    base = _annual_at(facts, tags, fy_end)
    if base is None or fy_end is None:
        return None, fy_end
    fy_dt = _parse_date(fy_end)
    quarters = _quarters_by_end(facts, tags)
    stubs = [
        q for q in quarters.values()
        if (_parse_date(q["start"]) - fy_dt).days >= -5
    ]
    if not stubs:
        return base, fy_end

    stub_sum, prior_sum = 0.0, 0.0
    latest_covered = fy_end
    for stub in sorted(stubs, key=lambda q: q["end"]):
        stub_end = _parse_date(stub["end"])
        counterpart = next(
            (
                q for q in quarters.values()
                if 330 <= (stub_end - _parse_date(q["end"])).days <= 430
            ),
            None,
        )
        if counterpart is None:
            return base, fy_end
        stub_sum += float(stub["val"])
        prior_sum += float(counterpart["val"])
        latest_covered = max(latest_covered, stub["end"])
    return base + stub_sum - prior_sum, latest_covered


def _quarterly_yoy(facts: Dict[str, Any], tags: List[str]) -> Optional[float]:
    """Latest discrete quarter vs. the same quarter one year earlier — a far
    fresher growth signal than fiscal-year-over-fiscal-year (Yahoo's
    revenueGrowth/earningsGrowth are quarterly YoY too, so this also keeps
    fallback growth comparable with normal-mode growth)."""
    quarters = _quarters_by_end(facts, tags)
    if len(quarters) < 2:
        return None
    ends = sorted(quarters, reverse=True)
    latest_val = float(quarters[ends[0]]["val"])
    latest_end = _parse_date(ends[0])
    for end in ends[1:]:
        if 330 <= (latest_end - _parse_date(end)).days <= 430:
            prior = float(quarters[end]["val"])
            return (latest_val - prior) / abs(prior) if prior != 0 else None
    return None


def _latest_instant(facts: Dict[str, Any], tags: List[str]) -> Optional[float]:
    """Most recent point-in-time (balance sheet) value across all candidate
    tags. Instants don't need period anchoring: the latest snapshot is the
    right one regardless of which tag carries it."""
    best_end, best_val = None, None
    for entries in _tag_entry_lists(facts, tags):
        for e in entries:
            if e.get("start") or e.get("val") is None:  # instants carry no start
                continue
            end = e.get("end") or ""
            if best_end is None or end > best_end:
                best_end, best_val = end, float(e["val"])
    return best_val


def _shares_outstanding(facts: Dict[str, Any]) -> Optional[float]:
    dei = facts.get("dei") or {}
    units = (dei.get("EntityCommonStockSharesOutstanding") or {}).get("units") or {}
    best_end, best_val = None, None
    for e in units.get("shares") or []:
        end = e.get("end") or ""
        if e.get("val") is not None and (best_end is None or end > best_end):
            best_end, best_val = end, float(e["val"])
    return best_val


def _sector_from_sic(sic: Any) -> Optional[str]:
    """Coarse SIC -> Yahoo-style sector. Only needs to be roughly right: it
    feeds the display and the cyclical-sector macro check, not the scores."""
    try:
        s = int(sic)
    except (TypeError, ValueError):
        return None
    if 2833 <= s <= 2836 or 3841 <= s <= 3851 or 8000 <= s <= 8099:
        return "Healthcare"
    if 3570 <= s <= 3579 or 3660 <= s <= 3699 or 3812 <= s <= 3827 or 7370 <= s <= 7379:
        return "Technology"
    if 100 <= s < 1000 or 2000 <= s < 2200:
        return "Consumer Defensive"
    if 1300 <= s < 1400 or 2900 <= s < 3000:
        return "Energy"
    if 1000 <= s < 1300 or 1400 <= s < 1500 or 2600 <= s < 2900 or 3300 <= s < 3400:
        return "Basic Materials"
    if 1500 <= s < 1800 or 3000 <= s < 4000 or 4000 <= s < 4800:
        return "Industrials"
    if 4800 <= s < 4900:
        return "Communication Services"
    if 4900 <= s < 5000:
        return "Utilities"
    if 5000 <= s < 6000 or 7000 <= s < 8000:
        return "Consumer Cyclical"
    if 6500 <= s < 6600 or s == 6798:
        return "Real Estate"
    if 6000 <= s < 6800:
        return "Financial Services"
    return None


def _ratio(a: Optional[float], b: Optional[float]) -> Optional[float]:
    return a / b if a is not None and b not in (None, 0) else None


def _growth(cur: Optional[float], prior: Optional[float]) -> Optional[float]:
    return (cur - prior) / abs(prior) if cur is not None and prior not in (None, 0) else None


def build_info_from_edgar(
    ticker: str,
    cik: str,
    last_price: Optional[float] = None,
    previous_close: Optional[float] = None,
    user_agent: str = "stock-research-script/1.0",
) -> Dict[str, Any]:
    """Rebuild a yfinance-.info-shaped dict from SEC filings. Returns {} if
    EDGAR is unreachable or the company has no usable annual facts."""
    cik10 = str(cik).zfill(10)
    try:
        subs = _get_json(f"https://data.sec.gov/submissions/CIK{cik10}.json", user_agent)
        company = _get_json(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik10}.json", user_agent)
    except Exception:
        return {}

    facts = company.get("facts") or {}
    name = company.get("entityName") or subs.get("name")
    if name and name.isupper():
        name = name.title()  # EDGAR shouts ("MICROSOFT CORPORATION"); the card shouldn't

    REVENUE_TAGS = [
        "RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues",
        "SalesRevenueNet", "RevenueFromContractWithCustomerIncludingAssessedTax",
        "Revenue", "RevenueFromContractsWithCustomers",
    ]
    NET_INCOME_TAGS = [
        "NetIncomeLoss", "ProfitLoss", "ProfitLossAttributableToOwnersOfParent",
    ]

    # Anchor everything to the newest fiscal year filed under ANY revenue or
    # net-income tag, then extend each flow to trailing-twelve-months where the
    # stub quarters net out cleanly (_ttm_at falls back to the anchored FY
    # value when they don't, so metrics can mix TTM and FY — approximately
    # right beats precisely stale).
    fy_end = _latest_fiscal_year_end(facts, REVENUE_TAGS + NET_INCOME_TAGS)

    revenue, rev_asof = _ttm_at(facts, REVENUE_TAGS, fy_end)
    net_income, ni_asof = _ttm_at(facts, NET_INCOME_TAGS, fy_end)
    operating_income, _ = _ttm_at(facts, [
        "OperatingIncomeLoss", "ProfitLossFromOperatingActivities",
    ], fy_end)
    gross_profit, _ = _ttm_at(facts, ["GrossProfit"], fy_end)
    dep_amort, _ = _ttm_at(facts, [
        "DepreciationDepletionAndAmortization", "DepreciationAndAmortization",
        "DepreciationAmortizationAndOther", "DepreciationAmortizationAndAccretionNet",
        "DepreciationAmortisationAndImpairmentLossReversalOfImpairmentLossRecognisedInProfitOrLoss",
    ], fy_end)
    ocf, _ = _ttm_at(facts, [
        "NetCashProvidedByUsedInOperatingActivities",
        "CashFlowsFromUsedInOperatingActivities",
    ], fy_end)
    capex, _ = _ttm_at(facts, [
        "PaymentsToAcquirePropertyPlantAndEquipment",
        "PurchaseOfPropertyPlantAndEquipment",
    ], fy_end)
    dividends_paid, _ = _ttm_at(facts, [
        "PaymentsOfDividends", "PaymentsOfDividendsCommonStock",
        "PaymentsOfOrdinaryDividends", "DividendsPaid",
    ], fy_end)

    # Growth: quarterly YoY when discrete quarters exist (fresher, and matches
    # how Yahoo defines revenueGrowth/earningsGrowth); FY-over-FY otherwise.
    revenue_growth = _quarterly_yoy(facts, REVENUE_TAGS)
    if revenue_growth is None:
        revenue_growth = _growth(
            _annual_at(facts, REVENUE_TAGS, fy_end),
            _annual_at(facts, REVENUE_TAGS, fy_end, offset_days=365),
        )
    earnings_growth = _quarterly_yoy(facts, NET_INCOME_TAGS)
    if earnings_growth is None:
        earnings_growth = _growth(
            _annual_at(facts, NET_INCOME_TAGS, fy_end),
            _annual_at(facts, NET_INCOME_TAGS, fy_end, offset_days=365),
        )

    equity = _latest_instant(facts, [
        "StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
        "Equity", "EquityAttributableToOwnersOfParent",
    ])
    cash = _latest_instant(facts, [
        "CashAndCashEquivalentsAtCarryingValue",
        "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
        "CashAndCashEquivalents",
    ])
    total_assets = _latest_instant(facts, ["Assets"])
    current_assets = _latest_instant(facts, ["AssetsCurrent", "CurrentAssets"])
    current_liabilities = _latest_instant(facts, ["LiabilitiesCurrent", "CurrentLiabilities"])
    inventory = _latest_instant(facts, ["InventoryNet", "Inventories"])

    lt_debt = _latest_instant(facts, ["LongTermDebtNoncurrent", "LongTermDebt", "NoncurrentPortionOfNoncurrentBorrowings"])
    st_debt = _latest_instant(facts, ["LongTermDebtCurrent", "DebtCurrent", "ShortTermBorrowings", "CurrentPortionOfLongtermBorrowings"])
    total_debt = lt_debt if st_debt is None else (st_debt if lt_debt is None else lt_debt + st_debt)

    # A company with no revenue AND no income facts gives the scorer nothing to
    # work with — report "no fallback" rather than an empty-but-plausible dict.
    if revenue is None and net_income is None:
        return {}

    shares = _shares_outstanding(facts)
    market_cap = last_price * shares if last_price is not None and shares else None
    enterprise_value = (
        market_cap + (total_debt or 0) - (cash or 0) if market_cap is not None else None
    )
    ebitda = (
        operating_income + dep_amort
        if operating_income is not None and dep_amort is not None
        else None
    )
    fcf = ocf - capex if ocf is not None and capex is not None else None
    eps = _ratio(net_income, shares)

    info: Dict[str, Any] = {
        "_source": "sec_edgar",
        "sec_data_asof": max(filter(None, [fy_end, rev_asof, ni_asof]), default=fy_end),
        "shortName": name,
        "longName": name,
        "sector": _sector_from_sic(subs.get("sic")),
        "industry": subs.get("sicDescription"),
        "currency": "USD",
        "currentPrice": last_price,
        "regularMarketPrice": last_price,
        "regularMarketPreviousClose": previous_close,
        "sharesOutstanding": shares,
        "marketCap": market_cap,
        "enterpriseValue": enterprise_value,
        "totalRevenue": revenue,
        "netIncomeToCommon": net_income,
        "ebitda": ebitda,
        "totalDebt": total_debt,
        "totalCash": cash,
        "freeCashflow": fcf,
        "operatingCashflow": ocf,
        "trailingPE": _ratio(last_price, eps) if eps and eps > 0 else None,
        "priceToSalesTrailing12Months": _ratio(market_cap, revenue),
        "priceToBook": _ratio(market_cap, equity),
        "enterpriseToRevenue": _ratio(enterprise_value, revenue),
        "enterpriseToEbitda": _ratio(enterprise_value, ebitda),
        "profitMargins": _ratio(net_income, revenue),
        "operatingMargins": _ratio(operating_income, revenue),
        "grossMargins": _ratio(gross_profit, revenue),
        "returnOnEquity": _ratio(net_income, equity),
        "returnOnAssets": _ratio(net_income, total_assets),
        "revenueGrowth": revenue_growth,
        "earningsGrowth": earnings_growth,
        "currentRatio": _ratio(current_assets, current_liabilities),
        # Quick ratio: (current assets - inventory) / current liabilities. A
        # missing inventory tag is treated as zero inventory — true for most
        # service/software filers, and matches how the ratio degenerates to
        # the current ratio for inventory-less companies anyway.
        "quickRatio": _ratio(
            current_assets - (inventory or 0) if current_assets is not None else None,
            current_liabilities,
        ),
        "debtToEquity": (
            total_debt / equity * 100 if total_debt is not None and equity not in (None, 0) else None
        ),
        # XBRL dividend payments are reported as positive outflows. The
        # chart-derived per-share yield (computed by the caller from actual
        # dividend events) overrides this cruder filings-based estimate.
        "dividendYield": _ratio(dividends_paid, market_cap),
        "payoutRatio": (
            dividends_paid / net_income
            if dividends_paid is not None and net_income is not None and net_income > 0
            else None
        ),
    }
    return {k: v for k, v in info.items() if v is not None}


def stooq_history(ticker: str, years: Optional[int] = None) -> pd.DataFrame:
    """Daily OHLCV history from Stooq (no key, no datacenter blocking) shaped
    like yfinance output: DatetimeIndex + Open/High/Low/Close/Volume columns.
    Returns an empty DataFrame when Stooq doesn't know the symbol."""
    symbol = ticker.lower().replace(".", "-") + ".us"
    try:
        res = requests.get(
            f"https://stooq.com/q/d/l/?s={symbol}&i=d",
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=EDGAR_TIMEOUT,
        )
        res.raise_for_status()
        text = res.text
        if not text.lstrip().startswith("Date"):
            return pd.DataFrame()
        df = pd.read_csv(io.StringIO(text), parse_dates=["Date"], index_col="Date")
    except Exception:
        return pd.DataFrame()
    if df.empty or "Close" not in df.columns:
        return pd.DataFrame()
    if years is not None:
        df = df[df.index >= df.index.max() - timedelta(days=int(years * 365.25))]
    return df
