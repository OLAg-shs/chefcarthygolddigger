import os
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
TWELVE_API_KEY = os.getenv("TWELVE_API_KEY")

BASE_URL = "https://api.twelvedata.com/time_series"

def get_data(symbol, interval="1h", outputsize=100):
    symbol_map = {
        "XAU/USD": "XAU/USD",
        "BTC/USD": "BTC/USD",
        "AAPL": "AAPL",
        "EUR/USD": "EUR/USD"
    }

    mapped_symbol = symbol_map.get(symbol, symbol)
    params = {
        "symbol": mapped_symbol,
        "interval": interval,
        "outputsize": outputsize,
        "apikey": TWELVE_API_KEY,
        "format": "JSON"
    }

    response = requests.get(BASE_URL, params=params)
    if response.status_code != 200:
        print(f"[ERROR] Failed to fetch data for {symbol}: HTTP {response.status_code}")
        return None

    data = response.json()

    if "values" not in data:
        print(f"[ERROR] Invalid data received for {symbol}: {data}")
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
    df = df.sort_values("time")  # Ascending time
    df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].astype(float)

    return df.reset_index(drop=True)
