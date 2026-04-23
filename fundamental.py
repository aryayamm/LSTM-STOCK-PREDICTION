import yfinance as yf

def get_fundamentals(ticker):
    try:
        stock = yf.Ticker(ticker)
        info  = stock.info
    except Exception:
        info = {}

    return {
        "EPS"           : info.get("trailingEps", 0) or 0,
        "ROE"           : info.get("returnOnEquity", 0) or 0,
        "ROA"           : info.get("returnOnAssets", 0) or 0,
        "DER"           : info.get("debtToEquity", 0) or 0,
        "PBV"           : info.get("priceToBook", 0) or 0,
        "PER"           : info.get("trailingPE", 0) or 0,
        "MarketCap"     : info.get("marketCap", 0) or 0,
        "DividendYield" : info.get("dividendYield", 0) or 0,
    }

def add_fundamentals(df, fundamentals):
    for key, value in fundamentals.items():
        df[key] = value
    return df