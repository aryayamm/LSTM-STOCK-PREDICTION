import warnings
warnings.filterwarnings("ignore")

from config import TICKER
from predictor import run_prediction
from notif import send_whatsapp
from database import init_db
from tracker import save_prediction, update_actual, get_accuracy_summary, get_history

def format_message(result, accuracy, history) -> str:
    t  = result["technical"]
    f  = result["fundamental"]
    s  = result["sentiment"]
    ticker = result["ticker"]

    direction_emoji = "🟢 UP" if result["direction"] == "UP" else "🔴 DOWN" if result["direction"] == "DOWN" else "🟡 SIDEWAYS"

    sector_text = ""
    for name, data in result["sector"].items():
        emoji = "🟢" if data["change"] > 0 else "🔴"
        sector_text += f"• {name} : {emoji} {data['change']:+.2f}%\n"

    local_news_text = "\n".join(s["local_news"]) if s["local_news"] else "No recent local news"
    macro_news_text = "\n".join(s["macro_news"]) if s["macro_news"] else "No recent macro news"

    accuracy_text = (
        f"🎯 Accuracy : {accuracy['accuracy']:.1f}% ({accuracy['correct']}/{accuracy['total']} correct)"
        if accuracy else "🎯 Accuracy : Not enough data yet"
    )

    history_text = ""
    for row in history:
        date, pred, actual, pred_dir, actual_dir, correct = row
        if correct is None:
            history_text += f"• {date} | Pred: Rp {pred:,.0f} | Actual: pending\n"
        else:
            tick = "✅" if correct else "❌"
            history_text += f"• {date} | Pred: Rp {pred:,.0f} | Actual: Rp {actual:,.0f} {tick}\n"

    div = f["DividendYield"]
    div_display = div / 100 if div > 1 else div

    return f"""🤖 LSTM Stock Prediction
📊 {ticker} (BBRI)

📍 Current Price  : Rp {result['current_price']:,.0f}
🔮 Predicted Price: Rp {result['predicted_price']:,.0f}
📈 Direction      : {direction_emoji}
📉 Change         : {result['change']:+.2f}%

📊 Technical
- RSI    : {t['rsi']:.1f} {'(Overbought ⚠️)' if t['rsi'] > 70 else '(Oversold 💡)' if t['rsi'] < 30 else '(Neutral)'}
- MACD   : {t['macd']:.2f} {'(Bullish 🟢)' if t['macd'] > 0 else '(Bearish 🔴)'}
- Trend  : {'🟢 Bullish' if t['ma7'] > t['ma30'] else '🔴 Bearish'}

🏦 Fundamental
- EPS    : {f['EPS']}
- ROE    : {f['ROE']:.2%}
- ROA    : {f['ROA']:.2%}
- DER    : {f['DER']}
- PBV    : {f['PBV']:.2f}
- PER    : {f['PER']:.2f}
- Div    : {div_display:.2%}

🌏 Sector Performance
{sector_text}
📰 Local News : {s['local_label']}
{local_news_text}

🌐 Macro News : {s['macro_label']}
{macro_news_text}

🎯 Overall Sentiment : {s['final_label']}

📊 Prediction History (last 7 days)
{history_text}
{accuracy_text}

⚠️ Not financial advice!"""

def main():
    print("=" * 40)
    print("🚀 Starting LSTM Stock Predictor")
    print("=" * 40)

    print("\n[0/3] 🗄️  Initializing database...")
    init_db()

    print("\n[1/3] 🤖 Running prediction...")
    result = run_prediction(TICKER)
    print("      ✅ Done!")

    print("\n[2/3] 🗄️  Saving to database...")
    update_actual(TICKER, result["current_price"])
    save_prediction(TICKER, result)  # ← pass full result now
    accuracy = get_accuracy_summary(TICKER)
    history  = get_history(TICKER, limit=7)
    print("      ✅ Done!")

    print("\n[3/3] 📲 Sending WhatsApp...")
    message = format_message(result, accuracy, history)
    print(message)
    send_whatsapp(message)
    print("\n✅ All done!")
    print("=" * 40)

main()