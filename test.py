import yfinance as yf
stock = yf.Ticker("AAPL")
expiry = stock.options[5]
chain = stock.option_chain(expiry)
print(chain.calls[['strike', 'volume', 'impliedVolatility']].head(20))