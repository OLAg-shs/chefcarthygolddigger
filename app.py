from flask import Flask, render_template
from utils.trading_bot import run_analysis

app = Flask(__name__)

@app.route('/')
def index():
    try:
        insights_text = run_analysis()
        market_insights = insights_text.strip().split('\n\n')
        return render_template("dashboard.html", market_insights=market_insights)
    except Exception as e:
        return render_template("dashboard.html", market_insights=[f"⚠️ Error: {e}"])

if __name__ == '__main__':
    app.run(debug=True)
