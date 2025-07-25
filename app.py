from flask import Flask, render_template
from utils.trading_bot import run_analysis
from datetime import datetime

app = Flask(__name__)

@app.route('/')
def index():
    try:
        insights = run_analysis()
        return render_template("dashboard.html", market_insights=insights, now=datetime.utcnow)
    except Exception as e:
        return render_template("dashboard.html", market_insights=[{"symbol": "Error", "text": f"⚠️ Error: {e}"}], now=datetime.utcnow)

if __name__ == '__main__':
    app.run(debug=True)
