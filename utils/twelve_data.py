import requests
import os
import pandas as pd

TWELVE_API_KEY = os.getenv("TWELVE_API_KEY")

def get_data(symbol, interval="1h", outputsize=100):
    url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval={interval}&outputsize={outputsize}&apikey={TWELVE_API_KEY}"
    response = requests.get(url)
    data = response.json()

    if "values" not in data:
        return None

    df = pd.DataFrame(data["values"])
    df = df.rename(columns={"datetime": "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")
    df.set_index("timestamp", inplace=True)

    df = df.astype(float)
    return df
