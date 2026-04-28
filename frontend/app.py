from flask import Flask, jsonify, request, render_template
from extraTrashTester import get_stock_assessment_for_html

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

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
