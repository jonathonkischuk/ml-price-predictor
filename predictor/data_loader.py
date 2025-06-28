import yfinance as yf
import pandas as pd
from pathlib import Path
import requests
import time


def load_stock_data(tickers, start='2020-01-01', end='2025-05-31'):
    data_dir = Path("data/raw")
    data_dir.mkdir(parents=True, exist_ok=True)

    for ticker in tickers:
        df = yf.download(ticker, start=start, end=end)
        df.to_csv(data_dir / f"{ticker}.csv")
        print(f"Downloaded {ticker}")


def get_crypto_data(symbol, days=365):
    url = f"https://api.coingecko.com/api/v3/coins/{symbol}/market_chart?vs_currency=usd&days={days}"
    res = requests.get(url)
    data = res.json()

    prices = pd.DataFrame(data['prices'], columns=['timestamp', 'price'])
    volumes = pd.DataFrame(data['total_volumes'], columns=['timestamp', 'volume'])
    df = prices.merge(volumes, on='timestamp')
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df = df.resample('1D').agg({'price': 'ohlc', 'volume': 'mean'})
    df.columns = df.columns.map('_'.join)
    return df