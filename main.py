import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
os.environ["PYTHONWARNINGS"] = "ignore"

import warnings
warnings.filterwarnings("ignore")

import yfinance as yf
from datetime import datetime
from config import TICKER, TICKERS
from predictor import run_prediction
from notif import send_whatsapp
from database import init_db, get_conn
from tracker import save_prediction, update_actual, get_accuracy_summary, get_history
from paper_trading import record_trade, get_portfolio_summary

def already_ran_today(ticker):
    try:
        conn = get_conn()
        c = conn.cursor()
        today = datetime.now().strftime("%Y-%m-%d")
        c.execute("SELECT id FROM predictions WHERE date = %s AND ticker = %s", (today, ticker))
        result = c.fetchone()
        conn.close()
        return result is not None
    except:
        return False

def format_message(result, accuracy, history, portfolio) -> str:
    t  = result["technical"]
    f  = result["fundamental"]
    s  = result["sentiment"]
    ticker   = result["ticker"]
    decision = result["decision"]
    conf     = result["confidence"]
    fg       = result.get("fear_greed", {})
    fg_value = fg.get("value", 50)
    fg_label = fg.get("label", "Neutral")
    fg_trend = fg.get("trend", "neutral")
    fg_emoji = "🟢" if fg_value > 60 else "🔴" if fg_value < 40 else "🟡"

    if decision == "NO_TRADE":
        buy_prob  = result['probs'].get('BUY', 0)
        sell_prob = result['probs'].get('SELL', 0)
        if buy_prob > sell_prob:
            decision_str = f"🟡 NO TRADE — BUY signal too weak ({buy_prob:.0%}, need 70%)"
        elif sell_prob > buy_prob:
            decision_str = f"🟡 NO TRADE — SELL signal too weak ({sell_prob:.0%}, need 70%)"
        else:
            decision_str = f"🟡 NO TRADE (uncertain)"
    elif decision == "BUY":
        decision_str = f"🟢 BUY (conf: {conf:.0%})"
    else:
        decision_str = f"🔴 SELL (conf: {conf:.0%})"

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

    if portfolio:
        port_text = f"""💼 Paper Portfolio
- Capital      : Rp {portfolio['capital']:,.0f}
- Total PnL    : {portfolio['total_pnl']:+.2f}%
- Total trades : {portfolio['total_trades']}
- Win rate     : {portfolio['win_rate']:.1f}%"""
    else:
        port_text = "💼 Paper Portfolio : No trades yet"

    div = f["DividendYield"]
    div_display = div / 100 if div > 1 else div

    return f"""🤖 LSTM + XGBoost Prediction
📊 {ticker}

🎯 TRADING SIGNAL
{decision_str}
- BUY    : {result['probs'].get('BUY', 0):.0%}
- SELL   : {result['probs'].get('SELL', 0):.0%}
- SKIP   : {result['probs'].get('NO_TRADE', 0):.0%}

📍 Current Price  : Rp {result['current_price']:,.0f}
🔮 LSTM Predicted : Rp {result['predicted_price']:,.0f}
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
- PBV    : {f['PBV']:.2f}
- PER    : {f['PER']:.2f}
- Div    : {div_display:.2%}

🌏 Sector
{sector_text}
📰 Local News : {s['local_label']}
{local_news_text}

🌐 Macro News : {s['macro_label']}
{macro_news_text}

🎯 Overall Sentiment : {s['final_label']}

😱 Fear & Greed  : {fg_emoji} {fg_value}/100 ({fg_label})
   5d Avg        : {fg.get('avg_5d', 50):.0f} — {fg_trend}

{port_text}

📊 Prediction History
{history_text}
{accuracy_text}

⚠️ Not financial advice!"""

def main():
    now = datetime.now()
    print("=" * 40)
    print(f"🚀 LSTM + XGBoost Predictor — {now.strftime('%H:%M:%S')}")
    print("=" * 40)

    print("\n[0] 🗄️  Initializing database...")
    init_db()

    for ticker in TICKERS:
        print(f"\n{'='*40}")
        print(f"📊 Processing {ticker}...")

        if now.hour < 16:
            # ── MORNING MODE — predict ──────────────
            print("  🌅 Morning run — predicting...")

            if already_ran_today(ticker):
                print(f"  ✅ Already predicted today, skipping!")
                continue

            print("\n  [1] 🤖 Running prediction...")
            result = run_prediction(ticker)
            print("      ✅ Done!")

            print("\n  [2] 🗄️  Saving to database...")
            save_prediction(ticker, result)
            accuracy  = get_accuracy_summary(ticker)
            history   = get_history(ticker, limit=7)
            print("      ✅ Done!")

            print("\n  [3] 💼 Recording paper trade...")
            record_trade(ticker, result)
            portfolio = get_portfolio_summary(ticker)
            print("      ✅ Done!")

            print("\n  [4] 📲 Sending WhatsApp...")
            message = format_message(result, accuracy, history, portfolio)
            print(message)
            send_whatsapp(message)
            print("      ✅ Done!")

        else:
            # ── AFTERNOON MODE — update actual ──────
            print("  🌆 Afternoon run — updating actual price...")

            try:
                stock        = yf.Ticker(ticker)
                hist         = stock.history(period="1d")
                actual_price = float(hist["Close"].iloc[-1])
                update_actual(ticker, actual_price)

                # Send confirmation WhatsApp
                send_whatsapp(f"📊 {ticker}\n✅ Actual price updated: Rp {actual_price:,.0f}")
                print(f"  ✅ Actual price: Rp {actual_price:,.0f}")
            except Exception as e:
                print(f"  ❌ Failed to update actual: {e}")

    print("\n✅ All done!")
    print("=" * 40)

main()