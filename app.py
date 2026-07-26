import threading

from flask import Flask, jsonify, request, render_template
from extraTrashTester import get_stock_assessment_for_html, get_macro_regime

app = Flask(__name__)


def _prewarm_macro_cache():
    # get_macro_regime() is cached for hours but costs several seconds cold.
    # Fetch it in the background at boot so the first visitor doesn't pay for it.
    try:
        get_macro_regime()
    except Exception:
        pass  # best-effort: the request path fetches it itself if this failed


threading.Thread(target=_prewarm_macro_cache, daemon=True).start()

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/health")
def health():
    # Cheap endpoint for uptime pings so the free instance never idles out.
    return "ok", 200

@app.route("/api/assessment")
def api_assessment():
    ticker = request.args.get("ticker", "").upper().strip()

    if not ticker:
        return jsonify({
            "success": False,
            "error": "Please enter a ticker."
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