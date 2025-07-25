import os
import requests
import pandas as pd
from dotenv import load_dotenv

# Load API key from .env or environment
load_dotenv()
TWELVE_API_KEY = os.getenv("TWELVE_API_KEY")


def get_data(symbol, interval="1h", outputsize=100):
    """
    Fetches OHLCV data from Twelve Data API.

    Args:
        symbol (str): e.g. "XAU/USD", "AAPL", "BTC/USD"
        interval (str): Time interval like '1h', '15min'
        outputsize (int): Number of candles to fetch (max 5000 for paid plans)

    Returns:
        pd.DataFrame or None if error
    """
    base_url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": symbol,
        "interval": interval,
        "outputsize": outputsize,
        "apikey": TWELVE_API_KEY
    }

    try:
        response = requests.get(base_url, params=params)
        data = response.json()

        if "status" in data and data["status"] == "error":
            print(f"[TwelveData ERROR] {data.get('message', 'Unknown error')}")
            return None

        if "values" not in data:
            print(f"[TwelveData ERROR] No values returned for {symbol}")
            return None

        df = pd.DataFrame(data["values"])
        df = df.rename(columns={
            "datetime": "timestamp",
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "volume": "volume"
        })

        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp")
        df.set_index("timestamp", inplace=True)

        # Convert numeric columns to float
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        return df.dropna()

    except Exception as e:
        print(f"[TwelveData Exception] Failed to fetch {symbol} data: {e}")
        return None
