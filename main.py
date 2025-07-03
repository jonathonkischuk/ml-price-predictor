# from predictor.data_loader import load_stock_data, get_crypto_data


# STOCKS = ['AMZN', 'GOOGL', 'PG', 'MSFT', 'TSM', 'IBM', 'META', 'RGTI', 'RITM', 'ET', 'EPD']
# CRYPTOS = {'jasmycoin': 'JASMY', 'render-token': 'RNDR'}

# #load_stock_data(STOCKS)

# for symbol, name in CRYPTOS.items():
#     df = get_crypto_data(symbol)
#     df.to_csv(f"data/raw/{name}.csv")


# ABOVE LOADS IN DATA FROM STOCKS/CRYPTO

# BELOW RUNS THE M MODEL
from predictor.feature_engineering import create_features
from predictor.model import train_model
import pandas as pd
import numpy as np
import os

ASSETS = ['AMZN', 'GOOGL', 'PG', 'MSFT', 'TSM', 'IBM', 'META', 'RGTI', 'RITM', 'ET', 'EPD', 'JASMY', 'RNDR']

os.makedirs("models", exist_ok=True)
all_results = {}

for ticker in ASSETS:
    print(f"\nProcessing {ticker}...")
    try:
        df = pd.read_csv(f"data/raw/{ticker}.csv")
        df = df.rename(columns=str.capitalize)
        df = create_features(df)

        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)

        print(f"{ticker} Class Balance: {df['target'].value_counts().to_dict()}")

        if df['target'].nunique() < 2 or len(df) < 300:
            print(f"Skipping {ticker}: Insufficient class balance or data length")
            continue

        _, fold_metrics = train_model(df, ticker)
        all_results[ticker] = fold_metrics

    except Exception as e:
        print(f"Error processing {ticker}: {e}")
