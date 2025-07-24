from flask import Flask, render_template
from utils.trading_bot import run_analysis_for_symbol

app = Flask(__name__)

@app.route("/")
def index():
    symbols = ["XAU/USD", "BTC/USD", "AAPL"]
    insights = {}
    for sym in symbols:
        result = run_analysis_for_symbol(sym)
        insights[sym] = result

    return render_template("dashboard.html", insights=insights)

if __name__ == "__main__":
    app.run(debug=True)
