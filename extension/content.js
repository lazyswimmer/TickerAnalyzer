// In-page card. Rendered inside a Shadow DOM so the host page's CSS can't break
// it and its CSS can't leak out. Shows a compact risk/quality summary, expandable
// to the full report — reusing the exact JSON the website's API returns.

(() => {
  if (window.__mraInitialized) return;
  window.__mraInitialized = true;

  const HOST_ID = "macro-risk-analyzer-host";
  let shadow = null;

  function esc(s) {
    return String(s == null ? "" : s).replace(/[&<>"]/g, c =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  }

  function level(score) {
    if (score <= 3) return "low";
    if (score <= 6) return "mod";
    return "high";
  }

  const STYLE = `
    :host { all: initial; color-scheme: light; }
    .panel {
      position: fixed; top: 18px; right: 18px; width: 380px; max-height: 90vh;
      overflow-y: auto; background: #fff; color: #1c2024; z-index: 2147483647;
      border: 1px solid #e4e7eb; border-radius: 12px; box-shadow: 0 10px 40px rgba(0,0,0,.18);
      font-family: -apple-system, "Segoe UI", Roboto, Arial, sans-serif; font-size: 14px;
      line-height: 1.5;
    }
    .hd { display: flex; justify-content: space-between; align-items: center;
      padding: 12px 16px; border-bottom: 1px solid #eef0f3; position: sticky; top: 0; background: #fff; }
    .hd .brand { font-weight: 700; font-size: 13px; color: #5b636b; letter-spacing: .02em; }
    .x { cursor: pointer; border: none; background: none; font-size: 20px; color: #9aa1a8; line-height: 1; }
    .bd { padding: 16px; }
    .muted { color: #5b636b; }
    .center { text-align: center; padding: 26px 16px; }
    .spinner { width: 26px; height: 26px; border: 3px solid #e4e7eb; border-top-color: #2563eb;
      border-radius: 50%; margin: 0 auto 12px; animation: mra-spin 0.9s linear infinite; }
    @keyframes mra-spin { to { transform: rotate(360deg); } }
    .ticker { font-size: 18px; font-weight: 700; }
    .verdict { color: #5b636b; margin-bottom: 14px; }
    .riskrow { display: flex; align-items: baseline; gap: 8px; }
    .score { font-size: 46px; font-weight: 800; line-height: 1; }
    .score span { font-size: 18px; color: #5b636b; font-weight: 600; }
    .risklabel { font-weight: 700; font-size: 13px; text-transform: uppercase; letter-spacing: .02em; margin-top: 4px; }
    .gauge { height: 9px; border-radius: 6px; background: #edf0f3; margin: 10px 0 6px; overflow: hidden; }
    .gauge > div { height: 100%; border-radius: 6px; }
    .quality { color: #5b636b; font-size: 13px; }
    .lvl-low { color: #1a9d6a; } .lvl-mod { color: #d98a00; } .lvl-high { color: #d6453d; }
    .bg-low { background: #1a9d6a; } .bg-mod { background: #d98a00; } .bg-high { background: #d6453d; }
    .sec { margin-top: 16px; }
    .h { font-weight: 700; font-size: 14px; margin-bottom: 8px; }
    .h.green { color: #1a9d6a; } .h.red { color: #d6453d; }
    .takeaway { background: #f0f5ff; border: 1px solid #d8e3fb; border-radius: 8px; padding: 12px 14px; }
    .bullets { margin: 0; padding-left: 18px; }
    .bullets li { margin-bottom: 6px; }
    .item { margin-bottom: 10px; }
    .item .t { font-weight: 600; }
    .item .d { color: #5b636b; font-size: 13px; }
    .expand { width: 100%; margin-top: 16px; padding: 10px; border: 1px solid #e4e7eb;
      background: #f6f7f9; color: #1c2024; font-family: inherit; border-radius: 8px;
      font-weight: 600; cursor: pointer; font-size: 14px; }
    .expand:hover { background: #eef0f3; }
    details { border-top: 1px solid #eef0f3; padding: 10px 0; }
    details > summary { cursor: pointer; font-weight: 600; list-style: none; }
    details > summary::-webkit-details-marker { display: none; }
    details > summary::after { content: " +"; color: #9aa1a8; }
    details[open] > summary::after { content: " \\2212"; }
    table { border-collapse: collapse; width: 100%; margin-top: 8px; }
    td, th { text-align: left; padding: 7px 8px; border-bottom: 1px solid #eef0f3; font-size: 13px; vertical-align: top; }
    th { color: #5b636b; font-weight: 600; } td:first-child, th:first-child { padding-left: 0; }
    .meaning { color: #5b636b; font-size: 12px; margin-top: 2px; }
    .cat { padding: 10px 0; border-bottom: 1px solid #eef0f3; }
    .cat:last-child { border-bottom: none; }
    .cat-head { display: flex; justify-content: space-between; font-weight: 600; }
    .cat-desc { color: #5b636b; font-size: 13px; margin-top: 3px; }
    .cat-notes { margin: 6px 0 0 16px; padding: 0; color: #5b636b; font-size: 13px; }
    .regime { background: #f2f6fb; border: 1px solid #d6e2f0; border-radius: 8px; padding: 9px 11px; }
    .err { color: #d6453d; font-weight: 600; }
    .foot { margin-top: 14px; color: #9aa1a8; font-size: 12px; text-align: center; }
    .foot a { color: #2563eb; text-decoration: none; font-weight: 600; }
    .foot a:hover { text-decoration: underline; }
  `;

  function ensurePanel() {
    let host = document.getElementById(HOST_ID);
    if (host) return shadow;
    host = document.createElement("div");
    host.id = HOST_ID;
    document.documentElement.appendChild(host);
    shadow = host.attachShadow({ mode: "open" });
    const style = document.createElement("style");
    style.textContent = STYLE;
    shadow.appendChild(style);
    const panel = document.createElement("div");
    panel.className = "panel";
    panel.innerHTML = `<div class="hd"><span class="brand">MACRO RISK ANALYZER · LITE</span>
      <button class="x" title="Close">&times;</button></div><div class="bd"></div>`;
    shadow.appendChild(panel);
    panel.querySelector(".x").addEventListener("click", () => host.remove());
    return shadow;
  }

  function body() { return shadow.querySelector(".bd"); }

  function showLoading(ticker) {
    ensurePanel();
    body().innerHTML = `<div class="center"><div class="spinner"></div>
      <div class="muted">Analyzing <strong>${esc(ticker)}</strong>…<br>this can take a moment.</div></div>`;
  }

  function showError(message) {
    ensurePanel();
    body().innerHTML = `<div class="center"><div class="err">${esc(message)}</div></div>`;
  }

  function items(list, empty) {
    if (!list || !list.length) return `<div class="item"><div class="d">${esc(empty)}</div></div>`;
    return list.map(it => `<div class="item"><div class="t">${esc(it.title)}</div>
      <div class="d">${esc(it.detail)}</div></div>`).join("");
  }

  function metricName(name, glossary) {
    const g = glossary && glossary[name];
    return `<div>${esc(name)}</div>` + (g ? `<div class="meaning">${esc(g)}</div>` : "");
  }

  function fullReport(data) {
    const g = data.metric_glossary || {};

    const cats = Object.entries(data.category_scores || {}).map(([cat, score]) => {
      const interp = (data.category_interpretations || {})[cat];
      const notes = ((data.summary_notes || {})[cat] || []).slice(0, 3);
      return `<div class="cat"><div class="cat-head"><span>${esc(cat)}</span>
        <span>${score === null ? "N/A" : esc(score) + "/10"}</span></div>
        ${interp ? `<div class="cat-desc">${esc(interp)}</div>` : ""}
        ${notes.length ? `<ul class="cat-notes">${notes.map(n => `<li>${esc(n)}</li>`).join("")}</ul>` : ""}</div>`;
    }).join("");

    const macro = data.macro_context || {};
    const macroNotes = (macro.stock_specific_notes || []).map(n => `<li>${esc(n)}</li>`).join("");
    const macroBlock = `<p class="regime">Current regime: ${esc(macro.regime || "unavailable")}</p>
      ${macroNotes ? `<ul class="cat-notes">${macroNotes}</ul>` : ""}`;

    const metricRows = Object.entries(data.metric_snapshot || {})
      .filter(([, v]) => v != null && v !== "" && v !== "N/A")
      .map(([k, v]) => `<tr><td>${metricName(k, g)}</td><td>${esc(v)}</td></tr>`).join("");

    return `<div class="full">
      <details><summary>Category breakdown</summary>${cats}</details>
      <details><summary>Macro context</summary>${macroBlock}</details>
      <details><summary>Key metrics</summary><table><tbody>${metricRows}</tbody></table></details>
    </div>`;
  }

  function showResult(data) {
    ensurePanel();
    const risk = data.generalized_risk_score;
    const lv = level(risk);
    const fit = Array.isArray(data.investor_fit)
      ? data.investor_fit
      : (data.investor_fit ? [data.investor_fit] : (data.key_takeaway ? [data.key_takeaway] : []));

    body().innerHTML = `
      <div class="ticker">${esc(data.ticker)} — Macro Risk Analysis</div>
      <div class="verdict">${esc(data.verdict || "")}</div>
      <div class="riskrow"><span class="score lvl-${lv}">${esc(risk)}<span>/10</span></span></div>
      <div class="risklabel lvl-${lv}">${esc(data.risk_label || "")} risk</div>
      <div class="gauge"><div class="bg-${lv}" style="width:${Math.max(0, Math.min(100, risk * 10))}%"></div></div>
      <div class="quality">Overall quality score: ${esc(data.overall_quality_score)}/10</div>

      <div class="sec takeaway"><div class="h">What this means for you</div>
        <ul class="bullets">${fit.map(b => `<li>${esc(b)}</li>`).join("")}</ul></div>

      <div class="sec"><div class="h green">Key strengths</div>${items(data.strengths, "No standout strengths identified.")}</div>
      <div class="sec"><div class="h red">Key risks</div>${items(data.risks, "No major risks flagged.")}</div>

      <button class="expand">Show full details ▾</button>
      ${fullReport(data)}
      <div class="foot">Generated in your browser by Macro Risk Analyzer Lite<br>
        <a href="https://tickeranalyzer.onrender.com/?ticker=${encodeURIComponent(data.ticker)}" target="_blank" rel="noopener">Get the full in-depth analysis of ${esc(data.ticker)} &rarr;</a></div>`;

    const full = body().querySelector(".full");
    const btn = body().querySelector(".expand");
    full.style.display = "none";
    btn.addEventListener("click", () => {
      const open = full.style.display !== "none";
      full.style.display = open ? "none" : "block";
      btn.textContent = open ? "Show full details ▾" : "Hide full details ▴";
    });
  }

  chrome.runtime.onMessage.addListener((msg) => {
    if (msg.type === "loading") showLoading(msg.ticker);
    else if (msg.type === "render") showResult(msg.data);
    else if (msg.type === "error") showError(msg.message);
  });
})();
