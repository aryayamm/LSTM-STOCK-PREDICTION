from technical import get_data
from fundamental import get_fundamentals, add_fundamentals
from sentiment import get_news_sentiment, add_sentiment
from sector import get_sector_data, add_sector, get_sector_summary
from lstm_signal import train_lstm, get_lstm_signals, LSTM_FEATURES
from xgb_decision import train_xgboost, get_xgb_decision, XGB_FEATURES
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

CONFIDENCE_THRESHOLD = 0.70

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

    df.dropna(inplace=True)

    # 5. Train LSTM
    print("Training LSTM...")
    from config import LOOK_BACK
    lstm_model, lstm_scaler, close_scaler = train_lstm(df)

    # 6. Generate LSTM signals
    print("Generating LSTM signals...")
    signals = get_lstm_signals(df, lstm_model, lstm_scaler, close_scaler)
    signal_df = pd.DataFrame(signals, index=df.index[LOOK_BACK:])
    df = df.join(signal_df)
    df.dropna(subset=["lstm_pred_price"], inplace=True)

    # 7. Train XGBoost
    print("Training XGBoost...")
    xgb_model, le = train_xgboost(df)

    # 8. Get final decision
    last_row = df.iloc[-1]
    current_price  = float(last_row["Close"])
    predicted_price = float(last_row["lstm_pred_price"])

    decision, confidence, probs = get_xgb_decision(last_row, xgb_model, le)

# Fix: if BUY or SELL is highest but below threshold → NO TRADE
    if decision != "NO_TRADE" and confidence < CONFIDENCE_THRESHOLD:
        decision = "NO_TRADE"
        confidence = probs.get("NO_TRADE", 0)

    # LSTM direction
    if predicted_price > current_price * 1.001:
        lstm_direction = "UP"
    elif predicted_price < current_price * 0.999:
        lstm_direction = "DOWN"
    else:
        lstm_direction = "SIDEWAYS"

    change = ((predicted_price - current_price) / current_price) * 100

    return {
        "ticker"          : ticker,
        "current_price"   : current_price,
        "predicted_price" : predicted_price,
        "direction"       : lstm_direction,
        "change"          : change,
        "decision"        : decision,
        "confidence"      : float(confidence),
        "probs"           : {k: float(v) for k, v in probs.items()},
        "technical": {
            "rsi"  : float(df["RSI"].iloc[-1]),
            "macd" : float(df["MACD"].iloc[-1]),
            "ma7"  : float(df["MA7"].iloc[-1]),
            "ma30" : float(df["MA30"].iloc[-1]),
        },
        "fundamental" : fundamentals,
        "sector"      : sector_summary,
        "sentiment": {
            "score"       : final_score,
            "local_label" : local_label,
            "macro_label" : macro_label,
            "final_label" : final_label,
            "local_news"  : local_results[:4],
            "macro_news"  : macro_results[:3],
        }
    }