import yfinance as yf
import pandas as pd
from config import PERIOD

SECTOR_TICKERS = {
    "IHSG" : "^JKSE",
}

BANK_NAMES = []

def get_sector_data(df_index):
    print("Fetching sector data...")

    sector_df = pd.DataFrame(index=df_index)

    for name, ticker in SECTOR_TICKERS.items():
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period=PERIOD)[["Close"]]
            hist = hist.reindex(df_index, method="ffill")

            hist[f"{name}_Return"] = hist["Close"].pct_change()
            hist[f"{name}_MA7"]    = hist["Close"].rolling(7).mean()
            hist[f"{name}_MA30"]   = hist["Close"].rolling(30).mean()

            sector_df[f"{name}_Return"] = hist[f"{name}_Return"]
            sector_df[f"{name}_MA7"]    = hist[f"{name}_MA7"]
            sector_df[f"{name}_MA30"]   = hist[f"{name}_MA30"]

            print(f"  ✅ {name} ({ticker}) fetched")
        except Exception as e:
            print(f"  ❌ {name} failed: {e}")
            sector_df[f"{name}_Return"] = 0
            sector_df[f"{name}_MA7"]    = 0
            sector_df[f"{name}_MA30"]   = 0

    # Average all 4 banks into one Banking sector column
    sector_df["Banking_Return"] = sector_df[[f"{b}_Return" for b in BANK_NAMES]].mean(axis=1)
    sector_df["Banking_MA7"]    = sector_df[[f"{b}_MA7"    for b in BANK_NAMES]].mean(axis=1)
    sector_df["Banking_MA30"]   = sector_df[[f"{b}_MA30"   for b in BANK_NAMES]].mean(axis=1)

    sector_df.fillna(0, inplace=True)
    return sector_df

def add_sector(df, sector_df):
    for col in sector_df.columns:
        df[col] = sector_df[col].values
    return df

def get_sector_summary():
    print("Fetching sector summary...")
    summary = {}

    # IHSG and LQ45
    for name in ["IHSG"]:
        ticker = SECTOR_TICKERS[name]
        try:
            hist = yf.Ticker(ticker).history(period="5d")
            current = hist["Close"].iloc[-1]
            prev    = hist["Close"].iloc[-2]
            change  = ((current - prev) / prev) * 100
            summary[name] = {"price": current, "change": change}
        except:
            summary[name] = {"price": 0, "change": 0}

    # Banking sector = average of 4 big banks
    bank_changes = []
    for name in BANK_NAMES:
        ticker = SECTOR_TICKERS[name]
        try:
            hist = yf.Ticker(ticker).history(period="5d")
            current = hist["Close"].iloc[-1]
            prev    = hist["Close"].iloc[-2]
            change  = ((current - prev) / prev) * 100
            bank_changes.append(change)
        except:
            pass

    summary["Banking (avg BBCA+BBNI+BMRI+BBRI)"] = {
        "price"  : 0,
        "change" : sum(bank_changes) / len(bank_changes) if bank_changes else 0
    }

    return summary