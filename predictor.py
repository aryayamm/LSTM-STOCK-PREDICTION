from technical import get_data
from fundamental import get_fundamentals, add_fundamentals
from sentiment import get_news_sentiment, add_sentiment
from sector import get_sector_data, add_sector, get_sector_summary
from model import train_and_predict
import warnings
warnings.filterwarnings("ignore")

def run_prediction(ticker: str) -> dict:
    # 1. Technical
    df = get_data(ticker)

    # 2. Fundamental
    fundamentals = get_fundamentals(ticker)
    df = add_fundamentals(df, fundamentals)

    # 3. Sector
    sector_df = get_sector_data(df.index)
    df = add_sector(df, sector_df)
    sector_summary = get_sector_summary()

    # 4. Sentiment
    final_score, local_results, macro_results, local_label, macro_label, final_label = get_news_sentiment(ticker)
    df = add_sentiment(df, final_score)

    # 5. Predict
    current_price, predicted_price = train_and_predict(df)
    change = ((predicted_price - current_price) / current_price) * 100

    if predicted_price > current_price * 1.001:
        direction = "UP"
    elif predicted_price < current_price * 0.999:
        direction = "DOWN"
    else:
        direction = "SIDEWAYS"

    return {
        "ticker"          : ticker,
        "current_price"   : current_price,
        "predicted_price" : predicted_price,
        "direction"       : direction,
        "change"          : change,
        "technical": {
            "rsi"  : df["RSI"].iloc[-1],
            "macd" : df["MACD"].iloc[-1],
            "ma7"  : df["MA7"].iloc[-1],
            "ma30" : df["MA30"].iloc[-1],
        },
        "fundamental"     : fundamentals,
        "sector"          : sector_summary,
        "sentiment": {
            "score"       : final_score,
            "local_label" : local_label,
            "macro_label" : macro_label,
            "final_label" : final_label,
            "local_news"  : local_results[:4],
            "macro_news"  : macro_results[:3],
        }
    }