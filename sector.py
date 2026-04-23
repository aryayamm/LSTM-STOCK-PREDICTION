import yfinance as yf
import pandas as pd
from config import PERIOD

SECTOR_TICKERS = {
    "IHSG"    : "^JKSE",
    "Nikkei"  : "^N225",
    "KOSPI"   : "^KS11",
    "HangSeng": "^HSI",
    "SGX"     : "^STI",
}

def get_sector_data(df_index):
    print("Fetching sector data...")
    sector_df = pd.DataFrame(index=df_index)

    for name, ticker in SECTOR_TICKERS.items():
        try:
            stock = yf.Ticker(ticker)
            hist  = stock.history(period=PERIOD)[["Close"]]

            # Remove timezone for clean reindexing
            if hist.index.tz is not None:
                hist.index = hist.index.tz_convert(None)
            if hasattr(df_index, 'tz') and df_index.tz is not None:
                df_index_naive = df_index.tz_convert(None)
            else:
                df_index_naive = df_index

            hist = hist.reindex(df_index_naive, method="ffill")
            hist.ffill(inplace=True)
            hist.bfill(inplace=True)

            # Skip if too many NaN
            nan_pct = hist["Close"].isna().mean()
            if nan_pct > 0.3:
                print(f"  ⚠️ {name} has {nan_pct:.0%} NaN — using zeros")
                raise ValueError("Too many NaN")

            hist[f"{name}_Return"] = hist["Close"].pct_change()
            hist[f"{name}_MA7"]    = hist["Close"].rolling(7).mean()
            hist[f"{name}_MA30"]   = hist["Close"].rolling(30).mean()

            sector_df[f"{name}_Return"] = hist[f"{name}_Return"].values
            sector_df[f"{name}_MA7"]    = hist[f"{name}_MA7"].values
            sector_df[f"{name}_MA30"]   = hist[f"{name}_MA30"].values

            print(f"  ✅ {name} ({ticker}) fetched")
        except Exception as e:
            print(f"  ❌ {name} failed: {e}")
            sector_df[f"{name}_Return"] = 0
            sector_df[f"{name}_MA7"]    = 0
            sector_df[f"{name}_MA30"]   = 0
    sector_df.fillna(0, inplace=True)
    return sector_df

def add_sector(df, sector_df):
    for col in sector_df.columns:
        df[col] = sector_df[col].values
    return df

def get_sector_summary():
    print("Fetching sector summary...")
    summary = {}

    for name, ticker in SECTOR_TICKERS.items():
        try:
            hist    = yf.Ticker(ticker).history(period="5d")
            current = hist["Close"].iloc[-1]
            prev    = hist["Close"].iloc[-2]
            change  = ((current - prev) / prev) * 100
            summary[name] = {"price": current, "change": change}
        except:
            summary[name] = {"price": 0, "change": 0}

    return summary