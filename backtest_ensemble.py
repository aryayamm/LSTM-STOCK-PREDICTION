import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from technical import get_data
from fundamental import get_fundamentals, add_fundamentals
from sector import get_sector_data, add_sector
from lstm_signal import train_lstm, get_lstm_signals, LSTM_FEATURES
from xgb_decision import train_xgboost, get_xgb_decision, XGB_FEATURES, create_labels
from config import LOOK_BACK

BACKTEST_TICKER = "TLKM.JK"
BACKTEST_DAYS   = 90
CONFIDENCE_THRESHOLD = 0.7  # only trade if confidence > 60%

def run_ensemble_backtest():
    print("=" * 50)
    print(f"📊 Ensemble Backtest — {BACKTEST_TICKER}")
    print("=" * 50)

    # Fetch data
    print("\n[1/5] Fetching data...")
    df = get_data(BACKTEST_TICKER)
    fundamentals = get_fundamentals(BACKTEST_TICKER)
    df = add_fundamentals(df, fundamentals)
    sector_df = get_sector_data(df.index)
    df = add_sector(df, sector_df)
    df["Sentiment"] = 0.5
    df.dropna(inplace=True)
    print(f"  ✅ Total rows: {len(df)}")

    train_end_idx = len(df) - BACKTEST_DAYS
    train_df = df.iloc[:train_end_idx].copy()
    test_df  = df.iloc[train_end_idx:].copy()

    # Train LSTM
    print("\n[2/5] Training LSTM...")
    lstm_model, lstm_scaler, close_scaler = train_lstm(train_df)

    # Generate LSTM signals for full dataset
    print("\n[3/5] Generating LSTM signals...")
    all_signals = get_lstm_signals(df, lstm_model, lstm_scaler, close_scaler)

    # Add signals to df (offset by LOOK_BACK)
    signal_df = pd.DataFrame(all_signals, index=df.index[LOOK_BACK:])
    df = df.join(signal_df)
    df.dropna(subset=["lstm_pred_price"], inplace=True)

    train_df = df.iloc[:train_end_idx - LOOK_BACK].copy()
    test_df  = df.iloc[train_end_idx - LOOK_BACK:].copy()

    # Train XGBoost
    print("\n[4/5] Training XGBoost...")
    xgb_model, le = train_xgboost(train_df)

    # Backtest
    print("\n[5/5] Running backtest...")
    results = []

    for i in range(len(test_df) - 1):
        row          = test_df.iloc[i]
        current_price = row["Close"]
        next_price    = test_df.iloc[i + 1]["Close"]

        try:
            decision, confidence, probs = get_xgb_decision(row, xgb_model, le)
        except:
            continue

        # Actual outcome
        actual_return = (next_price - current_price) / current_price * 100
        if actual_return > 0.5:
            actual = "BUY"
        elif actual_return < -0.5:
            actual = "SELL"
        else:
            actual = "NO_TRADE"

        # Only trade if confidence above threshold
        if confidence < CONFIDENCE_THRESHOLD:
            traded   = False
            decision = "NO_TRADE"
        else:
            traded = True

        correct = decision == actual

        # Simulate PnL
        pnl = 0
        if traded and decision == "BUY":
            pnl = actual_return
        elif traded and decision == "SELL":
            pnl = -actual_return

        results.append({
            "date"       : test_df.index[i].strftime("%Y-%m-%d"),
            "decision"   : decision,
            "actual"     : actual,
            "confidence" : round(confidence, 3),
            "traded"     : traded,
            "correct"    : correct,
            "pnl"        : round(pnl, 2),
            "lstm_signal": round(row["lstm_price_change"], 2),
            "probs"      : probs,
        })

        emoji = "✅" if correct else "❌"
        trade_str = f"[TRADED]" if traded else "[SKIP]  "
        print(f"  {test_df.index[i].strftime('%Y-%m-%d')} | {trade_str} | {decision:8} | Actual: {actual:8} | Conf: {confidence:.2f} | PnL: {pnl:+.2f}% {emoji}")

    # Summary
    results_df   = pd.DataFrame(results)
    total        = len(results_df)
    traded_rows  = results_df[results_df["traded"] == True]
    skipped      = total - len(traded_rows)

    if len(traded_rows) > 0:
        correct      = traded_rows["correct"].sum()
        accuracy     = (correct / len(traded_rows)) * 100
        total_pnl    = traded_rows["pnl"].sum()
        avg_pnl      = traded_rows["pnl"].mean()
        win_rate     = (traded_rows["pnl"] > 0).mean() * 100

        buy_rows  = traded_rows[traded_rows["decision"] == "BUY"]
        sell_rows = traded_rows[traded_rows["decision"] == "SELL"]
        buy_acc   = (buy_rows["correct"].mean() * 100) if len(buy_rows) > 0 else 0
        sell_acc  = (sell_rows["correct"].mean() * 100) if len(sell_rows) > 0 else 0
    else:
        accuracy = total_pnl = avg_pnl = win_rate = buy_acc = sell_acc = 0

    print("\n" + "=" * 50)
    print(f"📊 ENSEMBLE BACKTEST RESULTS — {BACKTEST_TICKER}")
    print("=" * 50)
    print(f"  Total days         : {total}")
    print(f"  Trades taken       : {len(traded_rows)} ({len(traded_rows)/total*100:.1f}%)")
    print(f"  Skipped (low conf) : {skipped}")
    print(f"  Direction accuracy : {accuracy:.1f}%")
    print(f"  BUY accuracy       : {buy_acc:.1f}%")
    print(f"  SELL accuracy      : {sell_acc:.1f}%")
    print(f"  Win rate           : {win_rate:.1f}%")
    print(f"  Total PnL          : {total_pnl:+.2f}%")
    print(f"  Avg PnL per trade  : {avg_pnl:+.2f}%")
    print("=" * 50)

    results_df.to_csv(f"backtest_ensemble_{BACKTEST_TICKER.replace('.', '_')}.csv", index=False)
    print(f"\n✅ Saved to backtest_ensemble_{BACKTEST_TICKER.replace('.', '_')}.csv")

    return results_df

run_ensemble_backtest()