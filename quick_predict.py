import warnings
warnings.filterwarnings("ignore")
import pandas as pd

from technical import get_data
from fundamental import get_fundamentals, add_fundamentals
from sentiment import get_news_sentiment, add_sentiment
from sector import get_sector_data, add_sector, get_sector_summary
from lstm_signal import train_lstm, get_lstm_signals
from xgb_decision import train_xgboost, get_xgb_decision
from config import LOOK_BACK

CONFIDENCE_THRESHOLD = 0.70
TICKERS = ["BBRI.JK", "BBCA.JK", "BMRI.JK"]

def quick_predict(ticker):
    print(f"\n{'='*40}")
    print(f"📊 Quick Predict — {ticker}")
    print(f"{'='*40}")

    # 1. Fetch all data
    print("[1] Fetching data...")
    df = get_data(ticker)
    fundamentals = get_fundamentals(ticker)
    df = add_fundamentals(df, fundamentals)
    df.index = df.index.tz_localize(None) if df.index.tz is None else df.index.tz_convert(None)
    sector_df = get_sector_data(df.index)
    df = add_sector(df, sector_df)
    final_score, local_results, macro_results, local_label, macro_label, final_label = get_news_sentiment(ticker)
    df = add_sentiment(df, final_score)
    df.dropna(inplace=True)

    # 2. Train LSTM
    print("[2] Training LSTM...")
    lstm_model, lstm_scaler = train_lstm(df)

    # 3. Generate signals
    print("[3] Generating signals...")
    signals   = get_lstm_signals(df, lstm_model, lstm_scaler)
    signal_df = pd.DataFrame(signals, index=df.index[LOOK_BACK:])
    df        = df.join(signal_df)
    df.dropna(subset=["lstm_pred_price"], inplace=True)

    # 4. Train XGBoost
    print("[4] Training XGBoost...")
    xgb_model, le = train_xgboost(df)

    # 5. Predict
    last_row      = df.iloc[-1]
    current_price = float(last_row["Close"])
    pred_price    = float(last_row["lstm_pred_price"])

    # Cap at 3%
    max_change = 0.03
    if pred_price > current_price * (1 + max_change):
        pred_price = current_price * (1 + max_change)
    elif pred_price < current_price * (1 - max_change):
        pred_price = current_price * (1 - max_change)

    change = ((pred_price - current_price) / current_price) * 100

    decision, confidence, probs = get_xgb_decision(last_row, xgb_model, le)
    if confidence < CONFIDENCE_THRESHOLD:
        decision = "NO_TRADE"

    if decision == "NO_TRADE":
        buy_prob  = probs.get("BUY", 0)
        sell_prob = probs.get("SELL", 0)
        if buy_prob > sell_prob:
            signal_str = f"🟡 NO TRADE — BUY too weak ({buy_prob:.0%}, need 70%)"
        elif sell_prob > buy_prob:
            signal_str = f"🟡 NO TRADE — SELL too weak ({sell_prob:.0%}, need 70%)"
        else:
            signal_str = "🟡 NO TRADE (uncertain)"
    elif decision == "BUY":
        signal_str = f"🟢 BUY (conf: {confidence:.0%})"
    else:
        signal_str = f"🔴 SELL (conf: {confidence:.0%})"

    direction = "UP" if pred_price > current_price * 1.001 else "DOWN" if pred_price < current_price * 0.999 else "SIDEWAYS"

    t   = last_row
    div = float(fundamentals.get("DividendYield", 0))
    div = div / 100 if div > 1 else div

    print(f"""
{'='*40}
🤖 RESULT — {ticker}
{'='*40}
🎯 SIGNAL     : {signal_str}
- BUY         : {probs.get('BUY', 0):.0%}
- SELL        : {probs.get('SELL', 0):.0%}
- SKIP        : {probs.get('NO_TRADE', 0):.0%}

📍 Current    : Rp {current_price:,.0f}
🔮 Predicted  : Rp {pred_price:,.0f}
📈 Direction  : {direction}
📉 Change     : {change:+.2f}%

📊 Technical
- RSI         : {float(t['RSI']):.1f}
- MACD        : {float(t['MACD']):.2f}
- Trend       : {'🟢 Bullish' if float(t['MA7']) > float(t['MA30']) else '🔴 Bearish'}

🏦 Fundamental
- EPS         : {fundamentals.get('EPS', 0)}
- ROE         : {fundamentals.get('ROE', 0):.2%}
- PBV         : {fundamentals.get('PBV', 0):.2f}
- PER         : {fundamentals.get('PER', 0):.2f}
- Div         : {div:.2%}

📰 Sentiment  : {final_label}
🌏 Sector Summary:""")

    sector_summary = get_sector_summary()
    for name, data in sector_summary.items():
        emoji = "🟢" if data["change"] > 0 else "🔴"
        print(f"  • {name}: {emoji} {data['change']:+.2f}%")

    print(f"""
📰 Local News : {local_label}""")
    for n in local_results[:3]:
        print(f"  {n}")
    print(f"""
🌐 Macro News : {macro_label}""")
    for n in macro_results[:3]:
        print(f"  {n}")

    print(f"\n⚠️  Not financial advice!")
    print("="*40)

    return {
        "ticker"    : ticker,
        "decision"  : decision,
        "confidence": confidence,
        "direction" : direction,
        "change"    : change,
    }

# Run for all tickers
results = []
for ticker in TICKERS:
    try:
        r = quick_predict(ticker)
        results.append(r)
    except Exception as e:
        print(f"❌ {ticker} failed: {e}")

# Summary at the end
print("\n" + "="*40)
print("📊 SUMMARY")
print("="*40)
for r in results:
    print(f"• {r['ticker']:10} | {r['decision']:8} | {r['direction']:8} | {r['change']:+.2f}%")