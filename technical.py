import yfinance as yf
from config import PERIOD
from fear_greed import get_fear_greed_history
from datetime import datetime, timedelta

def get_data(ticker: str):
    stock = yf.Ticker(ticker)
    df    = stock.history(period=PERIOD)
    df    = df[["Close", "Volume", "High", "Low"]]

    # Moving Averages
    df["MA7"]  = df["Close"].rolling(7).mean()
    df["MA30"] = df["Close"].rolling(30).mean()

    # RSI
    delta = df["Close"].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = -delta.clip(upper=0).rolling(14).mean()
    rs    = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))

    # MACD
    ema12        = df["Close"].ewm(span=12).mean()
    ema26        = df["Close"].ewm(span=26).mean()
    df["MACD"]   = ema12 - ema26
    df["Signal"] = df["MACD"].ewm(span=9).mean()
    df["MACD_Hist"] = df["MACD"] - df["Signal"]

    # Bollinger Bands
    df["BB_mid"]   = df["Close"].rolling(20).mean()
    df["BB_std"]   = df["Close"].rolling(20).std()
    df["BB_upper"] = df["BB_mid"] + 2 * df["BB_std"]
    df["BB_lower"] = df["BB_mid"] - 2 * df["BB_std"]
    df["BB_width"] = (df["BB_upper"] - df["BB_lower"]) / df["BB_mid"]

    # ROC
    df["ROC5"]  = df["Close"].pct_change(5) * 100
    df["ROC10"] = df["Close"].pct_change(10) * 100

    # Volume
    df["Volume_MA"]    = df["Volume"].rolling(20).mean()
    df["Volume_Ratio"] = df["Volume"] / df["Volume_MA"]

    # Stochastic
    low14        = df["Low"].rolling(14).min()
    high14       = df["High"].rolling(14).max()
    df["Stoch_K"] = (df["Close"] - low14) / (high14 - low14) * 100
    df["Stoch_D"] = df["Stoch_K"].rolling(3).mean()

    # Price distance from MA
    df["Price_MA7_dist"]  = (df["Close"] - df["MA7"]) / df["MA7"] * 100
    df["Price_MA30_dist"] = (df["Close"] - df["MA30"]) / df["MA30"] * 100

    # Daily return
    df["Daily_Return"] = df["Close"].pct_change() * 100

    # Fear & Greed — map historical values to dates
    fg_history = get_fear_greed_history(days=len(df) + 10)
    df.index = df.index.tz_convert(None) if df.index.tz else df.index
    df["FearGreed"] = df.index.map(
        lambda x: fg_history.get(x.strftime("%Y-%m-%d"), 0.5)
    )

    df.dropna(inplace=True)
    return df