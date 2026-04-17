import schedule
import time
from datetime import datetime
from config import TICKER
from predictor import run_prediction
from notif import send_whatsapp
from tracker import init_db, save_prediction, update_actual, get_accuracy_summary, get_history
from main import format_message
import warnings
warnings.filterwarnings("ignore")

TICKERS = ["BBRI.JK", "BBCA.JK", "BMRI.JK", "TLKM.JK"]  # add more anytime

def run_daily(tickers: list):
    print("=" * 40)
    print(f"🕘 Running daily predictions — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 40)

    init_db()

    for ticker in tickers:
        try:
            print(f"\n📊 Processing {ticker}...")
            result = run_prediction(ticker)

            # Save to DB
            update_actual(ticker, result["current_price"])
            save_prediction(ticker, result["predicted_price"], result["current_price"], result["direction"], result["change"])
            accuracy = get_accuracy_summary(ticker)
            history  = get_history(ticker, limit=7)

            # Send WhatsApp
            message = format_message(result, accuracy, history)
            send_whatsapp(message)
            print(f"  ✅ {ticker} done!")

        except Exception as e:
            print(f"  ❌ {ticker} failed: {e}")
            send_whatsapp(f"⚠️ Prediction failed for {ticker}\nError: {str(e)}")

    print("\n✅ All tickers done!")
    print("=" * 40)

def start_scheduler():
    # Run every day at 9:05 AM WIB (market opens 9AM)
    schedule.every().day.at("09:05").do(run_daily, tickers=TICKERS)

    # Also run at 3:35 PM WIB (market closes 3:30PM)
    schedule.every().day.at("15:35").do(run_daily, tickers=TICKERS)

    print("⏰ Scheduler started!")
    print("  → Runs at 09:05 WIB (market open)")
    print("  → Runs at 15:35 WIB (market close)")
    print("  → Press CTRL+C to stop\n")

    # Run immediately on start
    run_daily(TICKERS)

    while True:
        schedule.run_pending()
        time.sleep(60)

if __name__ == "__main__":
    start_scheduler()