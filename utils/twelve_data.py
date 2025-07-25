import requests
import os

API_KEY = os.getenv("TWELVE_API_KEY")
BASE_URL = "https://api.twelvedata.com/time_series"

def get_data(symbol: str, interval="1h", outputsize=100):
    params = {
        "symbol": symbol,
        "interval": interval,
        "outputsize": outputsize,
        "apikey": API_KEY,
        "format": "JSON"
    }

    response = requests.get(BASE_URL, params=params)

    if response.status_code != 200:
        raise Exception(f"TwelveData API error: {response.status_code} {response.text}")

    data = response.json()
    if "values" not in data:
        raise Exception(f"Invalid API response for {symbol}: {data}")

    return data["values"]
