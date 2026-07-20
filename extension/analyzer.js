// Fully client-side analysis engine — a JS port of the Python scoring.
// Runs in the extension service worker (CORS-exempt via host_permissions), so it
// needs no backend. Returns the SAME shape the card (content.js) already renders.

// ---------- formatting + glossary (mirrors extraTrashTester.py) ----------
const clamp = (v, lo = 0, hi = 10) => v == null ? null : Math.max(lo, Math.min(hi, v));
const r1 = v => Math.round(v * 10) / 10;

function num(v, d = 2) { return v == null || isNaN(v) ? "N/A" : Number(v).toLocaleString(undefined, { minimumFractionDigits: d, maximumFractionDigits: d }); }
function pct(v) { return v == null || isNaN(v) ? "N/A" : (v * 100).toFixed(1) + "%"; }
function money(v) {
  if (v == null || isNaN(v)) return "N/A";
  const a = Math.abs(v);
  if (a >= 1e12) return "$" + (v / 1e12).toFixed(2) + "T";
  if (a >= 1e9) return "$" + (v / 1e9).toFixed(2) + "B";
  if (a >= 1e6) return "$" + (v / 1e6).toFixed(2) + "M";
  return "$" + Math.round(v).toLocaleString();
}

const METRIC_GLOSSARY = {
  "Forward P/E": "Price versus next year's expected profit — higher means you pay more per dollar of earnings.",
  "Trailing P/E": "Price versus the past year's actual profit.",
  "EV/EBITDA": "Company value versus core operating profit (EBITDA); lower is generally cheaper.",
  "Price/Sales": "Price versus yearly revenue — useful when profits are thin.",
  "Revenue Growth": "How fast sales are growing year over year.",
  "Earnings Growth": "How fast profit is growing year over year.",
  "Operating Margin": "Profit left from core operations, as a share of revenue.",
  "Gross Margin": "Revenue left after the direct cost of making the product.",
  "Return on Equity": "Profit generated per dollar of shareholder money; higher is more efficient.",
  "FCF Margin": "Free cash flow (cash left after running and investing in the business) as a share of revenue.",
  "Net Debt / EBITDA": "Debt minus cash versus yearly operating profit — roughly how many years of profit it would take to repay debt; lower is safer.",
  "Current Ratio": "Short-term assets versus short-term bills — above 1 means it can cover near-term obligations.",
  "CAGR": "Compound annual growth rate — the average yearly return over the period.",
  "Annualized Volatility": "How much the price swings in a typical year; higher is a bumpier ride.",
  "Max Drawdown": "The worst peak-to-trough price drop over the period.",
  "Current RSI": "Relative Strength Index (0-100): above ~70 looks overbought, below ~30 oversold.",
  "Beta": "How much the stock moves relative to the market; above 1 is more volatile than the market.",
};

const JARGON = [
  ["EBITDA", "core operating profit before interest, taxes, and write-downs"],
  ["P/E", "price compared to profit"],
  ["PEG", "valuation weighed against the growth rate"],
  ["RSI", "a 0-100 momentum gauge"],
  ["CAGR", "average yearly return"],
  ["FCF", "free cash flow"],
  ["ROE", "return on equity"],
];
function humanize(t) {
  if (!t) return t;
  for (const [term, gloss] of JARGON) {
    const i = t.indexOf(term);
    if (i === -1) continue;
    const after = i + term.length;
    if (t.slice(after).trimStart().startsWith("(")) continue;
    t = t.slice(0, after) + " (" + gloss + ")" + t.slice(after);
  }
  return t;
}

// ---------- Yahoo data (cookie -> crumb -> quoteSummary -> chart) ----------
const UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36";

async function getCrumb() {
  // Visiting this seeds the Yahoo cookie in the browser jar; the SW then sends it
  // automatically (host_permissions cover the domains).
  await fetch("https://fc.yahoo.com", { headers: { "User-Agent": UA }, credentials: "include" }).catch(() => {});
  const res = await fetch("https://query1.finance.yahoo.com/v1/test/getcrumb",
    { headers: { "User-Agent": UA }, credentials: "include" });
  return (await res.text()).trim();
}

async function fetchQuoteSummary(ticker, crumb) {
  const modules = ["price", "summaryDetail", "defaultKeyStatistics", "financialData", "assetProfile"].join(",");
  const url = `https://query2.finance.yahoo.com/v10/finance/quoteSummary/${encodeURIComponent(ticker)}?modules=${modules}&crumb=${encodeURIComponent(crumb)}`;
  const res = await fetch(url, { headers: { "User-Agent": UA }, credentials: "include" });
  const json = await res.json();
  const result = json?.quoteSummary?.result?.[0];
  if (!result) throw new Error("No data for " + ticker);
  return result;
}

async function fetchCloses(ticker) {
  const url = `https://query1.finance.yahoo.com/v8/finance/chart/${encodeURIComponent(ticker)}?range=5y&interval=1d`;
  const res = await fetch(url, { headers: { "User-Agent": UA }, credentials: "include" });
  const json = await res.json();
  const closes = json?.chart?.result?.[0]?.indicators?.quote?.[0]?.close || [];
  return closes.filter(c => c != null);
}

const raw = x => (x && typeof x === "object" && "raw" in x) ? x.raw : (typeof x === "number" ? x : null);

// ---------- technical indicators from the close series ----------
function ema(arr, period) {
  const k = 2 / (period + 1);
  let e = arr[0];
  for (let i = 1; i < arr.length; i++) e = arr[i] * k + e * (1 - k);
  return e;
}
function technicals(closes) {
  if (closes.length < 50) return {};
  const last = closes[closes.length - 1];
  const sma200 = closes.slice(-200).reduce((a, b) => a + b, 0) / Math.min(200, closes.length);

  // RSI(14)
  const win = closes.slice(-15);
  let gain = 0, loss = 0;
  for (let i = 1; i < win.length; i++) { const d = win[i] - win[i - 1]; if (d >= 0) gain += d; else loss -= d; }
  const rs = loss === 0 ? 100 : (gain / 14) / (loss / 14);
  const rsi = loss === 0 ? 100 : 100 - 100 / (1 + rs);

  // MACD (12/26)
  const macdPositive = ema(closes.slice(-60), 12) > ema(closes.slice(-60), 26);

  // drawdowns + vol + cagr over the full series
  let peak = closes[0], maxDD = 0;
  for (const c of closes) { if (c > peak) peak = c; const dd = c / peak - 1; if (dd < maxDD) maxDD = dd; }
  const curDD = last / Math.max(...closes) - 1;
  const rets = [];
  for (let i = 1; i < closes.length; i++) rets.push(closes[i] / closes[i - 1] - 1);
  const mean = rets.reduce((a, b) => a + b, 0) / rets.length;
  const variance = rets.reduce((a, b) => a + (b - mean) ** 2, 0) / rets.length;
  const vol = Math.sqrt(variance) * Math.sqrt(252);
  const years = closes.length / 252;
  const cagr = years > 0 ? Math.pow(last / closes[0], 1 / years) - 1 : null;

  return {
    uptrend_above_200d: last > sma200, positive_MACD: macdPositive,
    current_RSI: rsi, current_drawdown: curDD,
    max_drawdown: maxDD, annualized_volatility: vol, cagr,
  };
}

// ---------- macro regime (FRED) + overlay ----------
async function fredLatest(code) {
  const url = `https://fred.stlouisfed.org/graph/fredgraph.csv?id=${code}&cosd=2018-01-01`;
  const res = await fetch(url, { headers: { "User-Agent": UA } });
  const text = await res.text();
  let last = null;
  for (const line of text.trim().split("\n").slice(1)) {
    const v = parseFloat(line.split(",")[1]);
    if (!isNaN(v)) last = v;
  }
  return last;
}
async function macroRegime() {
  const [fed, curve, hy] = await Promise.all([
    fredLatest("FEDFUNDS").catch(() => null),
    fredLatest("T10Y2Y").catch(() => null),
    fredLatest("BAMLH0A0HYM2").catch(() => null),
  ]);
  const flags = {
    curve_inverted: curve != null && curve < 0,
    rates_elevated: fed != null && fed >= 4.0,
    credit_stress: hy != null && hy >= 5.0,
  };
  const parts = [];
  if (fed != null) parts.push(`Fed funds ${fed.toFixed(2)}%`);
  if (curve != null) parts.push(`10y-2y ${curve >= 0 ? "+" : ""}${curve.toFixed(2)}%${curve < 0 ? " (inverted)" : ""}`);
  if (hy != null) parts.push(`HY credit spread ${hy.toFixed(2)}%`);
  return { fed, curve, hy, flags, label: parts.join("; ") || "Macro data unavailable" };
}
const CYCLICAL = new Set(["Consumer Cyclical", "Financial Services", "Basic Materials", "Industrials", "Energy", "Real Estate"]);
function macroOverlay(snap, balance, regime) {
  const f = regime.flags || {};
  let adj = 0; const notes = [];
  const beta = snap.beta, fpe = snap.forwardPE, nde = balance.net_debt_to_ebitda, sector = snap.sector;
  if (f.curve_inverted) {
    if (beta != null && beta > 1.2) { adj += 1; notes.push(`Yield curve is inverted (a recession signal) and this is a high-beta name (beta ${beta.toFixed(2)}); cyclical downside risk is elevated.`); }
    else if (CYCLICAL.has(sector)) { adj += 0.5; notes.push(`Yield curve is inverted (a recession signal) and ${sector} is a cyclical sector; earnings are more exposed to a slowdown.`); }
  }
  if (f.credit_stress && nde != null && nde > 3) { adj += 1; notes.push(`High-yield credit spreads are wide and leverage is high (net debt/EBITDA ${nde.toFixed(1)}x); refinancing risk is elevated.`); }
  if (f.rates_elevated && fpe != null && fpe > 30) { adj += 1; notes.push(`Policy rates are elevated and the stock is richly valued (forward P/E ${fpe.toFixed(0)}); high-multiple growth is rate-sensitive.`); }
  adj = Math.min(adj, 2.5);
  if (!notes.length) notes.push("No stock-specific macro risks flagged under the current regime.");
  return { adj, notes };
}

// ---------- scoring (ports of score_* in extraTrashTester.py) ----------
function scoreValuation(s) {
  let score = 5; const notes = [];
  const { forwardPE: fpe, trailingPE: tpe, enterpriseToEbitda: ev, pegRatio: peg } = s;
  if (fpe != null) { if (fpe < 15) { score += 1; notes.push("Forward P/E looks inexpensive on an absolute basis."); } else if (fpe > 35) { score -= 1; notes.push("Forward P/E looks expensive on an absolute basis."); } }
  if (peg != null) { if (peg > 2) { score -= 1; notes.push("PEG ratio suggests valuation may be high relative to growth."); } else if (peg > 0 && peg < 1.2) { score += 1; notes.push("PEG ratio appears reasonable relative to growth."); } }
  if (tpe != null && tpe < 0) { score -= 1; notes.push("Negative trailing P/E indicates current unprofitability."); }
  return [clamp(score), notes.length ? notes : ["Valuation appears roughly neutral from available metrics."]];
}
function scoreGrowth(s) {
  let score = 5; const notes = [];
  const { revenueGrowth: rg, earningsGrowth: eg, operatingMargins: om, grossMargins: gm, returnOnEquity: roe } = s;
  if (rg != null) { if (rg > 0.15) { score += 2; notes.push("Revenue growth is strong."); } else if (rg > 0.05) { score += 1; notes.push("Revenue growth is positive."); } else if (rg < 0) { score -= 2; notes.push("Revenue is declining."); } }
  if (eg != null) { if (eg > 0.15) { score += 1; notes.push("Earnings growth is strong."); } else if (eg < 0) { score -= 1; notes.push("Earnings are declining."); } }
  if (om != null) { if (om > 0.20) { score += 1; notes.push("Operating margin is strong."); } else if (om < 0) { score -= 2; notes.push("Operating margin is negative."); } }
  if (gm != null && gm > 0.50) { score += 1; notes.push("Gross margin suggests strong pricing power or software-like economics."); }
  if (roe != null) { if (roe > 0.15) { score += 1; notes.push("Return on equity is attractive."); } else if (roe < 0) { score -= 1; notes.push("Return on equity is negative."); } }
  return [clamp(score), notes.length ? notes : ["Growth and profitability appear mixed or incomplete."]];
}
function scoreBalance(b) {
  let score = 7; const notes = [];
  const { net_debt_to_ebitda: nde, current_ratio: cr, net_debt: nd } = b;
  if (nd != null && nd < 0) { score += 1; notes.push("Net cash position supports financial flexibility."); }
  else if (nde != null) { if (nde > 4) { score -= 3; notes.push("High net debt to EBITDA increases balance-sheet risk."); } else if (nde > 2.5) { score -= 2; notes.push("Moderate leverage deserves monitoring."); } else if (nde < 1) { score += 1; notes.push("Leverage appears low relative to EBITDA."); } }
  if (cr != null) { if (cr < 1) { score -= 2; notes.push("Current ratio below 1 may indicate liquidity pressure."); } else if (cr > 1.5) { score += 1; notes.push("Current ratio suggests adequate short-term liquidity."); } }
  return [clamp(score), notes.length ? notes : ["No major balance-sheet red flags detected."]];
}
function scoreEarnings(e) {
  let score = 7; const notes = [];
  const { fcf_margin: fcf, net_margin_calculated: nm, ocf_minus_net_income: ocf } = e;
  if (fcf != null) { if (fcf < 0) { score -= 3; notes.push("Free cash flow margin is negative."); } else if (fcf < 0.05) { score -= 1; notes.push("Free cash flow margin is thin."); } else if (fcf > 0.15) { score += 1; notes.push("Free cash flow margin is healthy."); } }
  if (nm != null) { if (nm < 0) { score -= 2; notes.push("Net margin is negative."); } else if (nm > 0.15) { score += 1; notes.push("Net margin is strong."); } }
  if (ocf != null) { if (ocf < 0) { score -= 2; notes.push("Operating cash flow is below net income."); } else { score += 1; notes.push("Operating cash flow supports reported earnings."); } }
  return [clamp(score), notes.length ? notes : ["Earnings quality appears acceptable from available data."]];
}
function scoreMarket(t) {
  let score = 5; const notes = [];
  if (t.uptrend_above_200d != null) { if (t.uptrend_above_200d) { score += 1; notes.push("Price is above its 200-day average."); } else { score -= 1; notes.push("Price is below its 200-day average."); } }
  if (t.positive_MACD != null) { if (t.positive_MACD) { score += 1; notes.push("MACD trend is positive."); } else { score -= 1; notes.push("MACD trend is not positive."); } }
  if (t.current_RSI != null) { if (t.current_RSI > 75) { score -= 1; notes.push("RSI is elevated; near-term overbought risk."); } else if (t.current_RSI < 30) { score -= 1; notes.push("RSI is depressed; momentum remains weak."); } }
  if (t.current_drawdown != null && t.current_drawdown < -0.30) { score -= 1; notes.push("Stock remains in a large drawdown."); }
  if (t.max_drawdown != null && t.max_drawdown < -0.60) { score -= 1; notes.push("Historical max drawdown has been severe."); }
  if (t.annualized_volatility != null) { if (t.annualized_volatility > 0.55) { score -= 2; notes.push("Historical volatility is very high."); } else if (t.annualized_volatility > 0.35) { score -= 1; notes.push("Historical volatility is elevated."); } }
  return [clamp(score), notes.length ? notes : ["Technical and market-positioning picture is neutral."]];
}
function scoreGovernance(g) {
  const vals = [g.auditRisk, g.boardRisk, g.compensationRisk, g.shareHolderRightsRisk, g.overallRisk].filter(v => typeof v === "number" && v > 0);
  if (!vals.length) return [5, ["Governance data unavailable or incomplete."]];
  const avg = vals.reduce((a, b) => a + b, 0) / vals.length;
  const notes = [`Average available governance risk score is ${avg.toFixed(1)}/10.`];
  if (avg >= 7) notes.push("Governance risk appears elevated."); else if (avg <= 3) notes.push("Governance risk appears relatively low.");
  return [clamp(11 - avg), notes];
}

// ---------- assemble (mirrors generate_scored_summary + build_*) ----------
function strong(v, th = 6.5) { return typeof v === "number" && v >= th; }

function buildRisksStrengths(scores, notes, macroNotes) {
  const firstNote = c => (notes[c] || []).find(n => n && !/unavailable/i.test(n)) || null;
  const rated = Object.entries(scores).filter(([, s]) => typeof s === "number");
  const strengths = [], risks = [];
  rated.slice().sort((a, b) => b[1] - a[1]).forEach(([c, s]) => { if (s >= 6.5 && strengths.length < 3) strengths.push({ title: c, detail: humanize(firstNote(c) || `Scores ${s}/10 in this category.`) }); });
  rated.slice().sort((a, b) => a[1] - b[1]).forEach(([c, s]) => { if (s <= 5.5 && risks.length < 3) risks.push({ title: c, detail: humanize(firstNote(c) || `Scores only ${s}/10 in this category.`) }); });
  for (const n of macroNotes) if (!/no stock-specific macro/i.test(n) && risks.length < 4) risks.push({ title: "Macro Exposure", detail: humanize(n) });
  if (!strengths.length && rated.length) { const [c, s] = rated.slice().sort((a, b) => b[1] - a[1])[0]; strengths.push({ title: c, detail: humanize(firstNote(c) || `Relatively strongest area (${s}/10).`) }); }
  if (!risks.length && rated.length) { const [c, s] = rated.slice().sort((a, b) => a[1] - b[1])[0]; risks.push({ title: c, detail: humanize(firstNote(c) || `Relatively weakest area (${s}/10).`) }); }
  return { strengths, risks };
}

function buildInvestorFit(ticker, scores, risk, riskLabel) {
  const aud = [];
  if (strong(scores["Growth / Profitability"])) aud.push("growth- and profitability-focused investors");
  if (strong(scores["Valuation"])) aud.push("value-conscious investors looking for a reasonable price");
  if (strong(scores["Balance Sheet"], 7) && risk <= 4) aud.push("conservative investors who prioritize stability");
  if (!aud.length) aud.push("investors comfortable with a mixed risk/reward profile");
  const audText = aud.length === 1 ? aud[0] : aud.slice(0, -1).join(", ") + " and " + aud[aud.length - 1];
  let caveat;
  if (risk >= 6.5) caveat = "Its higher risk makes it a weaker fit for cautious investors who are uncomfortable with big price swings or drops.";
  else if (risk >= 4) caveat = "With moderate risk, more cautious investors should keep positions small and keep an eye on the risks flagged below.";
  else caveat = "Its lower risk profile is relatively friendly to cautious investors.";
  return [`With ${riskLabel.toLowerCase()} risk (${risk}/10), ${ticker} may appeal most to ${audText}.`, caveat];
}

export async function analyzeTicker(ticker) {
  ticker = ticker.toUpperCase().trim();
  const crumb = await getCrumb();
  const [qs, closes, regime] = await Promise.all([
    fetchQuoteSummary(ticker, crumb),
    fetchCloses(ticker).catch(() => []),
    macroRegime().catch(() => ({ flags: {}, label: "Macro data unavailable" })),
  ]);

  const sd = qs.summaryDetail || {}, ks = qs.defaultKeyStatistics || {}, fd = qs.financialData || {}, pr = qs.price || {}, ap = qs.assetProfile || {};

  const snap = {
    shortName: pr.shortName || pr.longName, sector: ap.sector, industry: ap.industry,
    marketCap: raw(pr.marketCap) ?? raw(sd.marketCap),
    forwardPE: raw(sd.forwardPE) ?? raw(ks.forwardPE), trailingPE: raw(sd.trailingPE),
    enterpriseToEbitda: raw(ks.enterpriseToEbitda), priceToSalesTrailing12Months: raw(sd.priceToSalesTrailing12Months),
    pegRatio: raw(ks.pegRatio), revenueGrowth: raw(fd.revenueGrowth), earningsGrowth: raw(fd.earningsGrowth),
    operatingMargins: raw(fd.operatingMargins), grossMargins: raw(fd.grossMargins), returnOnEquity: raw(fd.returnOnEquity),
    beta: raw(sd.beta) ?? raw(ks.beta), profitMargins: raw(fd.profitMargins) ?? raw(ks.profitMargins),
  };

  const totalDebt = raw(fd.totalDebt), totalCash = raw(fd.totalCash), ebitda = raw(fd.ebitda), revenue = raw(fd.totalRevenue);
  const netDebt = (totalDebt != null && totalCash != null) ? totalDebt - totalCash : null;
  const balance = {
    net_debt: netDebt,
    net_debt_to_ebitda: (netDebt != null && ebitda) ? netDebt / ebitda : null,
    current_ratio: raw(fd.currentRatio),
  };
  const fcf = raw(fd.freeCashflow), ocf = raw(fd.operatingCashflow), netIncome = raw(ks.netIncomeToCommon);
  const earnings = {
    fcf_margin: (fcf != null && revenue) ? fcf / revenue : null,
    net_margin_calculated: snap.profitMargins,
    ocf_minus_net_income: (ocf != null && netIncome != null) ? ocf - netIncome : null,
  };
  const tech = technicals(closes);
  const gov = { auditRisk: ap.auditRisk, boardRisk: ap.boardRisk, compensationRisk: ap.compensationRisk, shareHolderRightsRisk: ap.shareHolderRightsRisk, overallRisk: ap.overallRisk };

  const [valuation, vNotes] = scoreValuation(snap);
  const [growth, gNotes] = scoreGrowth(snap);
  const [bal, bNotes] = scoreBalance(balance);
  const [earn, eNotes] = scoreEarnings(earnings);
  const [market, mNotes] = scoreMarket(tech);
  const [govScore, govNotes] = scoreGovernance(gov);

  const category_scores = {
    "Valuation": valuation, "Growth / Profitability": growth, "Balance Sheet": bal,
    "Earnings Quality": earn, "Market / Technical": market, "Governance": govScore,
  };
  const notes = {
    "Valuation": vNotes, "Growth / Profitability": gNotes, "Balance Sheet": bNotes,
    "Earnings Quality": eNotes, "Market / Technical": mNotes, "Governance": govNotes,
  };

  const available = Object.values(category_scores).filter(v => v != null);
  let quality = available.length ? available.reduce((a, b) => a + b, 0) / available.length : 5;
  let risk = 10 - quality;
  if (tech.annualized_volatility != null && tech.annualized_volatility > 0.50) risk += 1;
  if (tech.max_drawdown != null && tech.max_drawdown < -0.60) risk += 1;
  if (earnings.fcf_margin != null && earnings.fcf_margin < 0) risk += 1;
  if (balance.net_debt_to_ebitda != null && balance.net_debt_to_ebitda > 4) risk += 1;
  const overlay = macroOverlay(snap, balance, regime);
  risk += overlay.adj;
  risk = r1(clamp(risk)); quality = r1(clamp(quality));

  const riskLabel = risk <= 3 ? "Low" : risk <= 6 ? "Moderate" : risk <= 8 ? "High" : "Very High";
  const verdict = quality >= 7.5 ? "Attractive / strong profile" : quality >= 6 ? "Generally solid, but not risk-free" : quality >= 4.5 ? "Mixed / needs deeper research" : "Weak or speculative profile";

  // humanize the displayed notes
  for (const k in notes) notes[k] = notes[k].map(humanize);
  const { strengths, risks } = buildRisksStrengths(category_scores, notes, overlay.notes.map(humanize));

  const metric_snapshot = {
    "Company": snap.shortName, "Sector": snap.sector, "Industry": snap.industry,
    "Market Cap": money(snap.marketCap), "Forward P/E": num(snap.forwardPE), "Trailing P/E": num(snap.trailingPE),
    "EV/EBITDA": num(snap.enterpriseToEbitda), "Price/Sales": num(snap.priceToSalesTrailing12Months),
    "Revenue Growth": pct(snap.revenueGrowth), "Operating Margin": pct(snap.operatingMargins),
    "Gross Margin": pct(snap.grossMargins), "Return on Equity": pct(snap.returnOnEquity),
    "FCF Margin": pct(earnings.fcf_margin), "Net Debt / EBITDA": num(balance.net_debt_to_ebitda),
    "Beta": num(snap.beta), "CAGR": pct(tech.cagr), "Annualized Volatility": pct(tech.annualized_volatility),
    "Max Drawdown": pct(tech.max_drawdown), "Current RSI": num(tech.current_RSI),
  };

  return {
    success: true, ticker, verdict,
    overall_quality_score: quality, generalized_risk_score: risk, risk_label: riskLabel,
    investor_fit: buildInvestorFit(ticker, category_scores, risk, riskLabel),
    strengths, risks,
    metric_snapshot, metric_glossary: METRIC_GLOSSARY,
    category_scores, summary_notes: notes,
    macro_context: { regime: regime.label, stock_specific_notes: overlay.notes.map(humanize), risk_adjustment: r1(overlay.adj) },
    interpretation: "Computed entirely in your browser by Macro Risk Analyzer Lite. Generalized rule-based screen, not a buy/sell recommendation.",
  };
}
