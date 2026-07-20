# Chrome Web Store listing — Macro Risk Analyzer Lite

Copy/paste these into the Developer Dashboard fields when you create the new item.

---

## Item name
Macro Risk Analyzer Lite

## Summary (short description — max 132 chars)
Right-click any stock ticker to get an instant, in-browser risk & quality score. No account, no server, no setup.

## Category
Productivity

## Language
English (United States)

---

## Detailed description
Macro Risk Analyzer Lite turns any stock ticker on any web page into an instant,
plain-English risk read — without leaving the page.

Select a ticker (for example AAPL, $AAPL, or "Apple (AAPL)"), right-click, and choose
"Analyze … risk". A card appears with:

• A risk score (0–10) and an overall quality score
• "What this means for you" in plain language
• Key strengths and key risks
• An expandable report: category breakdown, macro context, and key metrics

Everything is computed right in your browser from public market data (Yahoo Finance)
and macroeconomic indicators (FRED). There is no account to create, no server to run,
and no setup — install it and start analyzing.

This is a lightweight screen, not investment advice. It uses a simplified rule-based
model, so its scores are directionally useful but are not a buy or sell recommendation.
Always do your own research.

---

## Single purpose (required field)
Show a risk and quality score for a stock ticker that the user selects on a web page.

---

## Permission justifications (required — one per permission)

**contextMenus**
Adds the right-click "Analyze '[ticker]' risk" menu item that appears when you select
text on a page.

**activeTab**
Lets the extension act only on the tab where you right-click a ticker, so it can show
the results card there. Access is granted by your click and limited to that tab.

**scripting**
Injects the results-card UI into the current page after you choose the menu item.

**Host permissions (query1/query2/fc.yahoo.com, fred.stlouisfed.org)**
The extension fetches public stock data from Yahoo Finance and public macroeconomic
data from FRED to compute the score. These are the only sites it contacts; it does not
access any other website's content.

**Remote code**
The extension does not use remote code. All logic is contained in the package.

---

## Data usage disclosures (Privacy tab)
- Does this item collect or use user data? **The extension collects no user data.**
- Personally identifiable info: **No**
- Health / financial / payment info: **No** (it reads public market data; it collects
  nothing about the user)
- Authentication info, location, web history, user activity, personal comms: **No**
- Sold or transferred to third parties: **No**
- Used for anything unrelated to the single purpose: **No**
- Used to determine creditworthiness / lending: **No**

Privacy policy URL: (paste the hosted URL of store/privacy-policy.html here)

---

## Screenshots to capture (1280×800 or 640×400, 1–5 total)
1. The results card open on a real page (e.g. a news article) after analyzing AAPL —
   show the risk score and "What this means for you".
2. The card with "Show full details" expanded (category breakdown + metrics).
3. The right-click menu showing "Analyze 'AAPL' risk".
4. (Optional) The toolbar popup.
