from predictor.data_loader import load_stock_data, get_crypto_data


STOCKS = ['AMZN', 'GOOGL', 'PG', 'MSFT', 'TSM', 'IBM', 'META', 'RGTI', 'RITM', 'ET', 'EPD']
CRYPTOS = {'jasmycoin': 'JASMY', 'render-token': 'RNDR'}

#load_stock_data(STOCKS)

for symbol, name in CRYPTOS.items():
    df = get_crypto_data(symbol)
    df.to_csv(f"data/raw/{name}.csv")
