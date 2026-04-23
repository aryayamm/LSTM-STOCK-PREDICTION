import numpy as np
import pandas as pd
import warnings
import os
import sys
import io

from technical import get_data
from fundamental import get_fundamentals, add_fundamentals
from sector import get_sector_data, add_sector
from lstm_signal import train_lstm, get_lstm_signals, LSTM_FEATURES
from xgb_decision import train_xgboost, get_xgb_decision, get_xgb_features
from config import LOOK_BACK
from fear_greed import get_fear_greed_history

warnings.filterwarnings("ignore")
sys.stdout = io.open("backtest_output.txt", "w", encoding="utf-8")

# ── CONFIG ────────────────────────────────────────
BACKTEST_TICKERS     = ["BBRI.JK", "BBCA.JK", "BMRI.JK"]
HOLD_DAYS_LIST       = [1]
BACKTEST_DAYS        = 90
CONFIDENCE_THRESHOLD = 0.70
STOP_LOSS            = 0.02
# ─────────────────────────────────────────────────
open("backtest_result.txt", "w").close()

def run_ensemble_backtest(BACKTEST_TICKER, HOLD_DAYS):
    print("=" * 50)
    print(f"📊 Ensemble Backtest — {BACKTEST_TICKER} (Hold {HOLD_DAYS}d)")
    print("=" * 50)

    # Fetch data
    print("\n[1/5] Fetching data...")
    df = get_data(BACKTEST_TICKER)
    fundamentals = get_fundamentals(BACKTEST_TICKER)
    df = add_fundamentals(df, fundamentals)
    sector_df = get_sector_data(df.index)
    df = add_sector(df, sector_df)

    # Shift Asian indices by 1 day (fix data leakage)
    for name in ["Nikkei", "KOSPI", "HangSeng", "SGX"]:
        for suffix in ["_Return", "_MA7", "_MA30"]:
            col = f"{name}{suffix}"
            if col in df.columns:
                df[col] = df[col].shift(1)

    df["Sentiment"] = 0.5

    # Fear & Greed historical data
    fg_history = get_fear_greed_history(days=len(df) + 10)
    df["FearGreed"] = df.index.map(
        lambda x: fg_history.get(
            (x.tz_convert(None) if x.tzinfo else x).strftime("%Y-%m-%d"),
            0.5
        )
    )

    df.dropna(inplace=True)
    print(f"  ✅ Total rows: {len(df)}")

    if len(df) < LOOK_BACK + BACKTEST_DAYS + 50:
        print(f"  ❌ Not enough data!")
        return None

    train_end_idx = len(df) - BACKTEST_DAYS

    # Train LSTM
    print("\n[2/5] Training LSTM...")
    train_df = df.iloc[:train_end_idx].copy()
    lstm_model, lstm_scaler = train_lstm(train_df)

    # Generate LSTM signals
    print("\n[3/5] Generating LSTM signals...")
    all_signals = get_lstm_signals(df, lstm_model, lstm_scaler)
    signal_df   = pd.DataFrame(all_signals, index=df.index[LOOK_BACK:])
    df          = df.join(signal_df)
    df.dropna(subset=["lstm_pred_price"], inplace=True)

    train_end_idx = len(df) - BACKTEST_DAYS
    train_df = df.iloc[:train_end_idx].copy()
    test_df  = df.iloc[train_end_idx:].copy()

    # Train XGBoost
    print("\n[4/5] Training XGBoost...")
    xgb_model, le, features = train_xgboost(train_df, BACKTEST_TICKER)
    

    # Backtest
    print("\n[5/5] Running backtest...")
    results = []
    i       = 0

    while i < len(test_df) - 1:
        row           = test_df.iloc[i]
        current_price = float(row["Close"])

        try:
            decision, confidence, probs = get_xgb_decision(row, xgb_model, le, features)
        except:
            i += 1
            continue

        if confidence < CONFIDENCE_THRESHOLD:
            decision = "NO_TRADE"

        if decision == "NO_TRADE":
            results.append({
                "date"      : test_df.index[i].strftime("%Y-%m-%d"),
                "decision"  : "NO_TRADE",
                "actual"    : "NO_TRADE",
                "confidence": round(confidence, 3),
                "traded"    : False,
                "correct"   : False,
                "pnl"       : 0,
                "hold_days" : 0,
                "hit_sl"    : False,
            })
            i += 1
            continue

        # Simulate holding
        entry_price  = current_price
        exit_price   = current_price
        actual_hold  = 0
        hit_stoploss = False

        for d in range(1, HOLD_DAYS + 1):
            if i + d >= len(test_df):
                break

            day_price   = float(test_df.iloc[i + d]["Close"])
            actual_hold = d

            if decision == "BUY":
                pnl_so_far = (day_price - entry_price) / entry_price
                if pnl_so_far < -STOP_LOSS:
                    exit_price   = day_price
                    hit_stoploss = True
                    break
            elif decision == "SELL":
                pnl_so_far = (entry_price - day_price) / entry_price
                if pnl_so_far < -STOP_LOSS:
                    exit_price   = day_price
                    hit_stoploss = True
                    break

            exit_price = day_price

        # Calculate PnL
        if decision == "BUY":
            pnl = (exit_price - entry_price) / entry_price * 100
        elif decision == "SELL":
            pnl = (entry_price - exit_price) / entry_price * 100
        else:
            pnl = 0

        actual_return = (exit_price - entry_price) / entry_price * 100
        if actual_return > 0.5:
            actual = "BUY"
        elif actual_return < -0.5:
            actual = "SELL"
        else:
            actual = "NO_TRADE"

        correct = decision == actual

        results.append({
            "date"      : test_df.index[i].strftime("%Y-%m-%d"),
            "decision"  : decision,
            "actual"    : actual,
            "confidence": round(confidence, 3),
            "traded"    : True,
            "correct"   : correct,
            "pnl"       : round(pnl, 2),
            "hold_days" : actual_hold,
            "hit_sl"    : hit_stoploss,
            "entry"     : entry_price,
            "exit"      : exit_price,
        })

        emoji  = "✅" if correct else "❌"
        sl_str = "🛑 SL" if hit_stoploss else f"{actual_hold}d"
        print(f"  {test_df.index[i].strftime('%Y-%m-%d')} | {decision:8} | Conf: {confidence:.2f} | PnL: {pnl:+.2f}% | {sl_str} {emoji}")

        i += actual_hold + 1

    # Summary
    results_df  = pd.DataFrame(results)
    total       = len(results_df)
    traded_rows = results_df[results_df["traded"] == True]
    skipped     = total - len(traded_rows)

    if len(traded_rows) > 0:
        correct      = traded_rows["correct"].sum()
        accuracy     = (correct / len(traded_rows)) * 100
        total_pnl    = traded_rows["pnl"].sum()
        avg_pnl      = traded_rows["pnl"].mean()
        win_rate     = (traded_rows["pnl"] > 0).mean() * 100
        stoploss_hit = traded_rows["hit_sl"].sum()
        avg_hold     = traded_rows["hold_days"].mean()
        buy_rows     = traded_rows[traded_rows["decision"] == "BUY"]
        sell_rows    = traded_rows[traded_rows["decision"] == "SELL"]
        buy_acc      = (buy_rows["correct"].mean() * 100) if len(buy_rows) > 0 else 0
        sell_acc     = (sell_rows["correct"].mean() * 100) if len(sell_rows) > 0 else 0
    else:
        accuracy = total_pnl = avg_pnl = win_rate = buy_acc = sell_acc = stoploss_hit = avg_hold = 0

    summary_path = "backtest_result.txt"

    with open(summary_path, "a", encoding="utf-8") as f:
        f.write("\n" + "=" * 50 + "\n")
        f.write(f"{BACKTEST_TICKER} — Hold {HOLD_DAYS}d\n")
        f.write("=" * 50 + "\n")
        f.write(f"Total days         : {total}\n")
        f.write(f"Trades taken       : {len(traded_rows)}\n")
        f.write(f"Skipped            : {skipped}\n")
        f.write(f"Direction accuracy : {accuracy:.1f}%\n")
        f.write(f"BUY accuracy       : {buy_acc:.1f}%\n")
        f.write(f"SELL accuracy      : {sell_acc:.1f}%\n")
        f.write(f"Win rate           : {win_rate:.1f}%\n")
        f.write(f"Stop loss hits     : {stoploss_hit}\n")
        f.write(f"Avg hold days      : {avg_hold:.1f}\n")
        f.write(f"Total PnL          : {total_pnl:+.2f}%\n")
        f.write(f"Avg PnL per trade  : {avg_pnl:+.2f}%\n")

    os.makedirs("backtest_results", exist_ok=True)
    results_df.to_csv(f"backtest_results/backtest_{BACKTEST_TICKER.replace('.', '_')}_hold{HOLD_DAYS}.csv", index=False)
    print(f"\n✅ Saved!")

    return results_df

# Run all combinations
for ticker in BACKTEST_TICKERS:
    for hold in HOLD_DAYS_LIST:
        run_ensemble_backtest(ticker, hold)

sys.stdout.close()
sys.stdout = sys.__stdout__
print("✅ Done! Check backtest_result.txt")