import yfinance as yf
from config import PERIOD

def get_data(ticker: str):
    stock = yf.Ticker(ticker)
    df = stock.history(period=PERIOD)
    df = df[["Close", "Volume", "High", "Low"]]

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
    ema12 = df["Close"].ewm(span=12).mean()
    ema26 = df["Close"].ewm(span=26).mean()
    df["MACD"]   = ema12 - ema26
    df["Signal"] = df["MACD"].ewm(span=9).mean()
    df["MACD_Hist"] = df["MACD"] - df["Signal"]  # ← new

    # Bollinger Bands
    df["BB_mid"]   = df["Close"].rolling(20).mean()
    df["BB_std"]   = df["Close"].rolling(20).std()
    df["BB_upper"] = df["BB_mid"] + 2 * df["BB_std"]
    df["BB_lower"] = df["BB_mid"] - 2 * df["BB_std"]
    df["BB_width"] = (df["BB_upper"] - df["BB_lower"]) / df["BB_mid"]  # ← new

    # Rate of Change (ROC) ← new
    df["ROC5"]  = df["Close"].pct_change(5) * 100   # 5 day momentum
    df["ROC10"] = df["Close"].pct_change(10) * 100  # 10 day momentum

    # Volume features ← new
    df["Volume_MA"] = df["Volume"].rolling(20).mean()
    df["Volume_Ratio"] = df["Volume"] / df["Volume_MA"]  # spike detection

    # Stochastic Oscillator ← new
    low14  = df["Low"].rolling(14).min()
    high14 = df["High"].rolling(14).max()
    df["Stoch_K"] = (df["Close"] - low14) / (high14 - low14) * 100
    df["Stoch_D"] = df["Stoch_K"].rolling(3).mean()

    # Price distance from MA ← new
    df["Price_MA7_dist"]  = (df["Close"] - df["MA7"]) / df["MA7"] * 100
    df["Price_MA30_dist"] = (df["Close"] - df["MA30"]) / df["MA30"] * 100

    # Daily return ← new
    df["Daily_Return"] = df["Close"].pct_change() * 100

    df.dropna(inplace=True)
    return df