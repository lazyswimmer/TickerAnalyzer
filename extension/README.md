# Macro Risk Analyzer Lite — Chrome extension

Highlight a stock ticker on any web page, right-click it, and get a risk & quality
score in a floating card — expandable to a category breakdown, macro context, and
key metrics.

**Everything runs in your browser.** No backend, no native host, no setup. The
extension fetches Yahoo Finance (cookie → crumb → `quoteSummary` + `chart`) and
FRED directly (CORS-exempt via `host_permissions`) and computes the scores in
[`analyzer.js`](analyzer.js).

## Lite vs. the full web app

This is the lightweight, zero-setup screen. It's a faithful port of the Python
scoring, but simplified: **no peer comparison** (that needs the bundled
financedatabase dataset) and earnings/balance metrics come from Yahoo's summary
fields rather than full statements — so scores are directionally right but won't
exactly match the full analysis.

For peer comparison and full-statement analysis, the plan is the **web app** (the
Flask `app.py` in the project root). v1 ships standalone with no link to it; once the
web app is deployed, add a link back in the card ([`content.js`](content.js)) and the
popup ([`popup.html`](popup.html)) and publish it as an extension update.

## Install (load unpacked)

1. Open `chrome://extensions` in Chrome.
2. Turn on **Developer mode** (top-right).
3. Click **Load unpacked** and select this `extension/` folder.
4. That's it — no backend needed.

## Use

1. On any page, select a ticker (e.g. `AAPL`, `$AAPL`, or `Apple (AAPL)`).
2. Right-click the selection → **Analyze "AAPL" risk**.
3. A card appears top-right with the risk score, quality score, what it means for
   you, and key strengths/risks.
4. Click **Show full details** to expand the category breakdown, macro context, and
   key metrics. Click **×** to close.

## Notes

- Injection isn't allowed on Chrome's own pages (`chrome://…`) or the Web Store.
- The card is rendered in a Shadow DOM, so it won't clash with the host page's CSS.
- Yahoo/FRED requests run in the service worker, which is why no server is needed.

## Files

- `manifest.json` — MV3 config (context menu, scripting, storage).
- `background.js` — service worker: context menu, ticker cleanup, runs the analysis.
- `analyzer.js` — the in-browser scoring engine (port of the Python scoring).
- `content.js` — the in-page card (compact + expandable report).
- `popup.html` — toolbar popup: what the extension does (static, no scripts).

## Before publishing to the Chrome Web Store

- (Future update) Once the web app is live, add its link to content.js + popup.html,
  and set the web app's "Get the extension" link (`templates/index.html`) to your
  Web Store listing URL.
- `host_permissions` is already narrowed to the Yahoo + FRED hosts the analyzer
  needs — no broad wildcards to justify in review.
- Provide proper icon sizes, screenshots, and a privacy policy (the extension only
  fetches public market data and stores nothing personal).
