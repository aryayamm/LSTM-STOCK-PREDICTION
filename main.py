import warnings
warnings.filterwarnings("ignore")

from config import TICKER
from technical import get_data
from fundamental import get_fundamentals, add_fundamentals
from sentiment import get_news_sentiment, add_sentiment
from sector import get_sector_data, add_sector, get_sector_summary
from model import train_and_predict
from notif import send_whatsapp
from tracker import init_db, save_prediction, update_actual, get_accuracy_summary, get_history

def predict_direction(current, predicted):
    if predicted > current * 1.001:
        return "🟢 UP"
    elif predicted < current * 0.999:
        return "🔴 DOWN"
    else:
        return "🟡 SIDEWAYS"

def main():
    print("=" * 40)
    print("🚀 Starting LSTM Stock Predictor")
    print("=" * 40)

    print("\n[0/7] 🗄️ Initializing database...")
    init_db()

    print("\n[1/7] 📈 Fetching technical data...")
    df = get_data()
    print("      ✅ Done!")

    print("\n[2/7] 🏦 Fetching fundamental data...")
    fundamentals = get_fundamentals()
    df = add_fundamentals(df, fundamentals)
    print("      ✅ Done!")

    print("\n[3/7] 🌏 Fetching sector data...")
    sector_df = get_sector_data(df.index)
    df = add_sector(df, sector_df)
    sector_summary = get_sector_summary()
    print("      ✅ Done!")

    print("\n[4/7] 📰 Fetching news & sentiment...")
    final_score, local_results, macro_results, local_label, macro_label, final_label = get_news_sentiment()
    df = add_sentiment(df, final_score)
    print("      ✅ Done!")

    print("\n[5/7] 🤖 Training LSTM model...")
    current_price, predicted_price = train_and_predict(df)
    direction = predict_direction(current_price, predicted_price)
    change = ((predicted_price - current_price) / current_price) * 100
    print("      ✅ Done!")

    print("\n[6/7] 🗄️ Saving prediction & updating actual...")
    update_actual(TICKER, current_price)
    save_prediction(TICKER, predicted_price, current_price, direction, change)
    accuracy = get_accuracy_summary(TICKER)
    history  = get_history(TICKER, limit=7)
    print("      ✅ Done!")

    print("\n[7/7] 📲 Sending WhatsApp message...")

    rsi  = df["RSI"].iloc[-1]
    macd = df["MACD"].iloc[-1]
    ma7  = df["MA7"].iloc[-1]
    ma30 = df["MA30"].iloc[-1]

    sector_text = ""
    for name, data in sector_summary.items():
        emoji = "🟢" if data["change"] > 0 else "🔴"
        sector_text += f"• {name} : {emoji} {data['change']:+.2f}%\n"

    if accuracy:
        accuracy_text = f"🎯 Accuracy : {accuracy['accuracy']:.1f}% ({accuracy['correct']}/{accuracy['total']} correct)"
    else:
        accuracy_text = "🎯 Accuracy : Not enough data yet"

    history_text = ""
    for row in history:
        date, pred, actual, pred_dir, actual_dir, correct = row
        if correct is None:
            history_text += f"• {date} | Pred: Rp {pred:,.0f} | Actual: pending\n"
        else:
            tick = "✅" if correct else "❌"
            history_text += f"• {date} | Pred: Rp {pred:,.0f} | Actual: Rp {actual:,.0f} {tick}\n"

    local_news_text = "\n".join(local_results[:4]) if local_results else "No recent local news"
    macro_news_text = "\n".join(macro_results[:3]) if macro_results else "No recent macro news"

    message = f"""🤖 LSTM Stock Prediction
📊 {TICKER} (BBRI)

📍 Current Price  : Rp {current_price:,.0f}
🔮 Predicted Price: Rp {predicted_price:,.0f}
📈 Direction      : {direction}
📉 Change         : {change:+.2f}%

📊 Technical
- RSI    : {rsi:.1f} {'(Overbought ⚠️)' if rsi > 70 else '(Oversold 💡)' if rsi < 30 else '(Neutral)'}
- MACD   : {macd:.2f} {'(Bullish 🟢)' if macd > 0 else '(Bearish 🔴)'}
- Trend  : {'🟢 Bullish' if ma7 > ma30 else '🔴 Bearish'}

🏦 Fundamental
- EPS    : {fundamentals['EPS']}
- ROE    : {fundamentals['ROE']:.2%}
- ROA    : {fundamentals['ROA']:.2%}
- DER    : {fundamentals['DER']}
- PBV    : {fundamentals['PBV']:.2f}
- PER    : {fundamentals['PER']:.2f}
- Div    : {(fundamentals['DividendYield'] / 100 if fundamentals['DividendYield'] > 1 else fundamentals['DividendYield']):.2%}

🌏 Sector Performance
{sector_text}
📰 Local News : {local_label}
{local_news_text}

🌐 Macro News : {macro_label}
{macro_news_text}

🎯 Overall Sentiment : {final_label}

📊 Prediction History (last 7 days)
{history_text}
{accuracy_text}

⚠️ Not financial advice!"""

    print(message)
    send_whatsapp(message)
    print("\n✅ All done!")
    print("=" * 40)

main()