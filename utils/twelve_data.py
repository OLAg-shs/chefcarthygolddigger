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

    try:
        response = requests.get(BASE_URL, params=params)
        response.raise_for_status()
        data = response.json()

        if "values" not in data:
            print(f"[WARN] No 'values' in response for {symbol}: {data}")
            return None

        df = pd.DataFrame(data["values"])
        df = df.rename(columns={
            "datetime": "datetime",
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "volume": "volume"
        })

        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.sort_values("datetime")
        df.set_index("datetime", inplace=True)

        # Convert numeric columns
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        return df.dropna()

    except Exception as e:
        print(f"[ERROR] Failed to fetch data for {symbol}: {e}")
        return None
