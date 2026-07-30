import re
import threading
import time

from flask import Flask, jsonify, request, render_template
from extraTrashTester import (
    get_stock_assessment_for_html,
    get_macro_regime,
    get_sec_cik_map,
)

app = Flask(__name__)

# Ticker autocomplete list, built from the SEC's official company_tickers.json
# (every SEC-registered company: ticker + name, ordered by market cap, so the
# search gets a free relevance tiebreak). Cached in memory and refreshed daily.
_TICKER_LIST_CACHE = {"data": None, "fetched_at": 0.0}
_TICKER_LIST_TTL = 24 * 60 * 60
_TICKER_LIST_MAX = 8000  # cap the payload; nobody autocompletes the micro-cap tail


def _ticker_list():
    now = time.time()
    if (
        _TICKER_LIST_CACHE["data"] is not None
        and now - _TICKER_LIST_CACHE["fetched_at"] < _TICKER_LIST_TTL
    ):
        return _TICKER_LIST_CACHE["data"]

    def display_name(title):
        # SEC titles are inconsistently cased ("MICROSOFT CORP" vs "Apple Inc.");
        # title-case the all-caps ones so the dropdown doesn't shout.
        title = str(title)
        return title.title() if title.isupper() else title

    df = get_sec_cik_map()
    records = [
        {"t": row.ticker, "n": display_name(row.title)}
        for row in df.head(_TICKER_LIST_MAX).itertuples()
    ]
    _TICKER_LIST_CACHE["data"] = records
    _TICKER_LIST_CACHE["fetched_at"] = now
    return records


def _prewarm_caches():
    # All cached for hours but cost seconds-to-minutes cold; fetch in the
    # background at boot so the first visitor doesn't pay for them. The dummy
    # assessment matters most: the true first analysis on a cold instance can
    # exceed the gunicorn timeout (session handshakes + peer universe +
    # statement parsing on a small shared CPU), so run one before users do.
    # It also fills the .info cache with mega-cap peers most analyses share.
    for warm in (get_macro_regime, _ticker_list, lambda: get_stock_assessment_for_html("AAPL")):
        try:
            warm()
        except Exception:
            pass  # best-effort: the request path fetches these itself if needed


threading.Thread(target=_prewarm_caches, daemon=True).start()

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/health")
def health():
    # Cheap endpoint for uptime pings so the free instance never idles out.
    return "ok", 200

@app.route("/api/tickers")
def api_tickers():
    # Compact [{t: ticker, n: name}] list for the search autocomplete. Matching
    # happens client-side; this is fetched once per day per browser.
    try:
        data = _ticker_list()
    except Exception:
        return jsonify([]), 503

    response = jsonify(data)
    response.headers["Cache-Control"] = "public, max-age=86400"
    return response


@app.route("/api/assessment")
def api_assessment():
    ticker = request.args.get("ticker", "").upper().strip()

    if not ticker:
        return jsonify({
            "success": False,
            "error": "Please enter a ticker."
        }), 400

    # Cheap format gate: real US tickers are 1-10 chars of letters/digits with
    # optional . or - (BRK.B, FTV-PA). Anything else skips Yahoo entirely.
    if not re.fullmatch(r"[A-Z0-9.\-]{1,10}", ticker):
        return jsonify({
            "success": False,
            "ticker": ticker,
            "error": f'"{ticker}" doesn\'t look like a stock ticker. '
                     "Try a symbol like AAPL or MSFT.",
        }), 400

    try:
        result = get_stock_assessment_for_html(ticker)
        return jsonify(result)

    except Exception as exc:
        return jsonify({
            "success": False,
            "ticker": ticker,
            "error": str(exc)
        }), 500

if __name__ == "__main__":
    app.run(debug=True)