import os
import requests
import pandas as pd

API_KEY = os.getenv("TWELVE_API_KEY")

def get_data(symbol, interval="1h", outputsize=200):
    url = f"https://api.twelvedata.com/time_series"
    params = {
        "symbol": symbol,
        "interval": interval,
        "apikey": API_KEY,
        "outputsize": outputsize,
        "format": "JSON"
    }

    try:
        response = requests.get(url, params=params)
        data = response.json()
        if "values" not in data:
            print(f"⚠️ API error for {symbol}: {data.get('message')}")
            return None

        df = pd.DataFrame(data["values"])
        df = df.rename(columns={
            "datetime": "time",
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "volume": "volume"
        })
        df["time"] = pd.to_datetime(df["time"])
        df = df.astype({
            "open": float,
            "high": float,
            "low": float,
            "close": float,
            "volume": float
        })
        df = df.sort_values("time").reset_index(drop=True)
        return df
    except Exception as e:
        print(f"❌ Error fetching data for {symbol}: {e}")
        return None
