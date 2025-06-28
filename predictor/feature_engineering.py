import pandas as pd
import numpy as np


def create_features(df):
    df['return'] = df['Close'].pct_change()
    df['volume'] = df['Volume'].pct_change()
    df['rsi'] = df['Close'].rolling(14).apply(lambda x: np.mean(x[-1] > x[:-1]), raw=True)
    df['volatility'] = df['Close'].rolling(5).std()
    df['momentum'] = df['Close'] - df['Close'].shift(10)

    df['target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)
    df.dropna(inplace=True)
    return df