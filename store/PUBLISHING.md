# Publishing Macro Risk Analyzer Lite — step-by-step

Everything needed to get the extension live on the Chrome Web Store. Follow top to
bottom; check each box as you go.

Your upload file: **`store/macro-risk-analyzer-lite-v0.1.0.zip`**
Your listing copy: **`store/listing.md`** (copy/paste source for every text field)
Your privacy page: **`store/privacy-policy.html`** (host it, step 1B)

---

## PHASE 0 — One-time developer account setup

### 0A. Open the developer console
1. Go to **https://chrome.google.com/webstore/devconsole**
2. Sign in with the Google account you want to own the extension.
   - Use an account you'll keep long-term — the extension lives under it.

### 0B. Accept terms + pay the fee
3. Read and accept the **Developer Agreement** when prompted.
4. Pay the **one-time $5 USD registration fee** (Google Payments — credit/debit card).
   - This is charged once ever, not per extension.

### 0C. Set your publisher identity
5. Go to **Account** (left sidebar) → set a **Publisher display name** (what users
   see as the author, e.g. your name or a brand).
6. Add and **verify a contact email** (Google sends a verification link). You cannot
   publish until this email is verified.

---

## PHASE 1 — Prep the assets (do before creating the item)

### 1A. Confirm you have the zip
- File: `store/macro-risk-analyzer-lite-v0.1.0.zip`
- It contains manifest.json, background.js, analyzer.js, content.js, popup.html,
  icon.png — all at the zip's top level (NOT inside a subfolder). Already built for you.

### 1B. Host the privacy policy and copy its URL  ← REQUIRED
The Web Store requires a public privacy policy URL. Pick ONE easy way to host
`store/privacy-policy.html`:

- **GitHub Pages** (free): create a public repo, add the file, enable Pages in repo
  Settings → Pages, and use the resulting `https://<you>.github.io/...` URL.
- **Netlify Drop** (free, no account needed): go to app.netlify.com/drop and drag the
  HTML file in; it gives you a public URL instantly.
- **Google Sites / Notion (public page)**: paste the text into a public page.

➡️ **Save the final public URL** — you'll paste it in Phase 4.

### 1C. Capture screenshots  ← REQUIRED (at least 1)
Screenshots MUST be exactly **1280×800** or **640×400** pixels (PNG or JPEG).

To create them:
1. Load the extension unpacked so you can use it:
   - `chrome://extensions` → enable **Developer mode** → **Load unpacked** →
     select the `extension/` folder.
2. Open a normal web page (e.g. a news article that mentions a stock).
3. Select a ticker like `AAPL`, right-click → **Analyze "AAPL" risk**.
4. When the card appears, take a screenshot (Windows: **Win+Shift+S**).
5. Because the capture won't be exactly 1280×800, resize/pad it to that exact size:
   - Easiest: open the image at **photopea.com** (free, in-browser) → Image → Canvas
     Size → set 1280×800 → export as PNG. Or use Paint: resize the canvas to 1280×800.
6. Recommended shots (1–5 total):
   - The card open on the page (risk score + "What this means for you").
   - The card with **Show full details** expanded.
   - The right-click menu showing the "Analyze ... risk" item.

➡️ **Save the 1280×800 image files** — you'll upload them in Phase 3.

### 1D. (Optional) Store icon
The listing uses a 128×128 store icon. Your `extension/icon.png` is already 128×128,
so you can upload that same file if the dashboard asks for a store icon.

---

## PHASE 2 — Create the item and upload

1. In the console, click **+ New Item** (top right).
2. In the upload dialog, **drag in** `macro-risk-analyzer-lite-v0.1.0.zip`
   (or click to browse). Click **Upload**.
3. Wait for it to process. If it errors, it's almost always a zip-structure issue
   (files must be at the root of the zip) — the provided zip is already correct.
4. You now land on the item's editing screens with tabs across the top:
   **Store listing**, **Privacy practices**, **Distribution**. Fill them in order.

---

## PHASE 3 — Store listing tab

Open `store/listing.md` alongside this and copy each value in.

1. **Product name**: `Macro Risk Analyzer Lite` (may be pre-filled from the manifest).
2. **Summary**: paste the short description (the ≤132-char line).
3. **Description**: paste the detailed description.
4. **Category**: choose **Productivity**.
5. **Language**: **English (United States)**.
6. **Store icon**: upload `extension/icon.png` (128×128) if requested.
7. **Screenshots**: upload the 1280×800 images from step 1C (drag them in).
8. **Promo tiles** (small 440×280, marquee 1400×560): **optional — skip** for v1.
9. Click **Save draft** (top right).

---

## PHASE 4 — Privacy practices tab  ← where most first-timers get stuck

Everything you need is pre-written in `store/listing.md`. Fill in:

1. **Single purpose**: paste the single-purpose sentence.
2. **Permission justifications**: for EACH item listed, paste the matching
   justification from listing.md:
   - `contextMenus`
   - `activeTab`
   - `scripting`
   - Host permission (the Yahoo + FRED domains)
   - **Remote code**: select **"No, I am not using remote code"** (all code is in the
     package).
3. **Data usage**: 
   - Check the box certifying you do **NOT** collect or use user data (the extension
     collects nothing — see listing.md for the exact answers).
   - Answer **No** to selling/transferring data and to unrelated uses.
4. **Privacy policy URL**: paste the URL you hosted in step 1B.
5. Tick the **certification** checkbox that your disclosures are accurate.
6. Click **Save draft**.

---

## PHASE 5 — Distribution tab + submit

1. **Visibility / Distribution**: choose one:
   - **Public** — listed and searchable by everyone. Pick this for a normal launch.
   - **Unlisted** — anyone with the link can install, but it's not searchable. Good if
     you want to test the live listing quietly first.
   - **Private** — only specified testers.
2. **Regions**: leave **all regions** unless you want to limit.
3. **Pricing**: **Free**.
4. Click **Save draft**, then click **Submit for review** (top right).
5. Confirm any final dialog.

---

## PHASE 6 — After you submit

- **Review time**: usually a few hours to a few days. You'll get an email on approval
  or rejection.
- **If rejected**: the email states the reason. Fix it, bump nothing needed, re-submit
  the same item. (Common reasons don't apply here — permissions are minimal and
  justified — but read the note if it happens.)
- **Once approved**: your public URL is `https://chromewebstore.google.com/detail/<id>`.
  Copy it — that's the "Get the extension" link for your website later.

---

## LATER — v1.1 update (once the web app is live)

1. In `extension/content.js` and `extension/popup.html`, add the link to your deployed
   web app.
2. In `extension/manifest.json`, bump `"version"` to `0.1.1`.
3. Re-zip the `extension/` files (same 6 files, at the zip root).
4. In the console, open the item → **Package** → **Upload new package** → submit.
5. Installed users update automatically within a day.

(Ask Claude to "rebuild the zip for v1.1" and it'll regenerate the package for you.)
