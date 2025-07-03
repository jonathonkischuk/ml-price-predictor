import pandas as pd
import numpy as np

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def create_features(df):
    df['return'] = df['Close'].pct_change()
    df['volume_change'] = df['Volume'].pct_change()
    df['rsi'] = compute_rsi(df['Close'], 14)
    df['volatility'] = df['Close'].rolling(5).std()
    df['momentum'] = df['Close'] - df['Close'].shift(10)

    df['ema_10'] = df['Close'].ewm(span=10, adjust=False).mean()
    df['ema_50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['bb_upper'] = df['Close'].rolling(20).mean() + 2 * df['Close'].rolling(20).std()
    df['bb_lower'] = df['Close'].rolling(20).mean() - 2 * df['Close'].rolling(20).std()

    ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema_12 - ema_26

    for lag in [1, 2, 3]:
        df[f'return_lag{lag}'] = df['return'].shift(lag)
        df[f'rsi_lag{lag}'] = df['rsi'].shift(lag)

    df['rolling_mean_10'] = df['Close'].rolling(10).mean()
    df['rolling_std_10'] = df['Close'].rolling(10).std()
    df['momentum_ratio'] = df['momentum'] / df['Close']
    df['rsi_vol'] = df['rsi'] * df['volatility']
    df['day_of_week'] = pd.to_datetime(df['Date']).dt.dayofweek

    df['price_vs_ema_10'] = (df['Close'] - df['ema_10']) / df['ema_10']
    df['price_vs_ema_50'] = (df['Close'] - df['ema_50']) / df['ema_50']

    df['future_return'] = df['Close'].shift(-3) / df['Close'] - 1
    df['target'] = np.where(df['future_return'] > 0.01, 1, 0)

    df.dropna(inplace=True)
    return df
