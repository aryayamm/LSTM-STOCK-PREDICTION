import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from technical import get_data
from fundamental import get_fundamentals, add_fundamentals
from sentiment import get_news_sentiment, add_sentiment
from sector import get_sector_data, add_sector
from model import FEATURE_COLS, build_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping
from config import LOOK_BACK, EPOCHS, BATCH_SIZE

# ── CONFIG ────────────────────────────────────────
BACKTEST_TICKER = "BBRI.JK"
BACKTEST_DAYS   = 120
# ─────────────────────────────────────────────────

def prepare_backtest_data(df, train_end_idx):
    scaler = MinMaxScaler()
    train_data = df[FEATURE_COLS].iloc[:train_end_idx]
    scaler.fit(train_data)

    scaled = scaler.transform(df[FEATURE_COLS])

    X, y = [], []
    for i in range(LOOK_BACK, train_end_idx - 1):
        X.append(scaled[i - LOOK_BACK:i])
        next_close = df["Close"].iloc[i + 1]
        curr_close = df["Close"].iloc[i]
        y.append(1 if next_close > curr_close else 0)

    X, y = np.array(X), np.array(y)
    return X, y, scaler

def run_backtest():
    print("=" * 50)
    print(f"📊 Backtesting {BACKTEST_TICKER}")
    print(f"   Backtest days : {BACKTEST_DAYS}")
    print("=" * 50)

    # Get full dataset
    print("\n[1/4] Fetching data...")
    df = get_data(BACKTEST_TICKER)
    fundamentals = get_fundamentals(BACKTEST_TICKER)
    df = add_fundamentals(df, fundamentals)

    sector_df = get_sector_data(df.index)
    df = add_sector(df, sector_df)
    df["Sentiment"] = 0.5
    df.dropna(inplace=True)

    print(f"  ✅ Total rows : {len(df)}")

    train_end_idx = len(df) - BACKTEST_DAYS
    print(f"  ✅ Train rows : {train_end_idx}")
    print(f"  ✅ Test rows  : {BACKTEST_DAYS}")

    # Prepare data
    print("\n[2/4] Preparing data...")
    X_train, y_train, scaler = prepare_backtest_data(df, train_end_idx)

    from collections import Counter
    print(f"  Class balance: {Counter(y_train)}")

    # Class weights to fix DOWN bias
    classes = np.unique(y_train)
    weights = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weight = dict(zip(classes, weights))
    print(f"  Class weights: {class_weight}")

    # Train model
    print("\n[3/4] Training model...")
    model = build_model((X_train.shape[1], X_train.shape[2]))
    model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.1,
        class_weight=class_weight,
        callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)],
        verbose=1
    )

    # Backtest day by day
    print("\n[4/4] Running backtest...")
    scaled_full = scaler.transform(df[FEATURE_COLS])

    results = []
    for i in range(BACKTEST_DAYS):
        idx = train_end_idx + i

        if idx >= len(df):
            break

        window = scaled_full[idx - LOOK_BACK:idx]
        window = np.expand_dims(window, axis=0)

        prob          = model.predict(window, verbose=0)[0][0]
        current_price = df["Close"].iloc[idx]
        prev_price    = df["Close"].iloc[idx - 1]

        pred_direction   = "UP" if prob > 0.5 else "DOWN"
        actual_direction = "UP" if current_price > prev_price * 1.001 else "DOWN" if current_price < prev_price * 0.999 else "SIDEWAYS"
        correct          = pred_direction == actual_direction
        error            = abs(current_price - prev_price) / prev_price * 100

        results.append({
            "date"            : df.index[idx].strftime("%Y-%m-%d"),
            "prob_up"         : round(float(prob), 3),
            "pred_direction"  : pred_direction,
            "actual_direction": actual_direction,
            "current_price"   : round(float(current_price), 0),
            "prev_price"      : round(float(prev_price), 0),
            "correct"         : correct,
            "error_pct"       : round(error, 2),
        })

        print(f"  {df.index[idx].strftime('%Y-%m-%d')} | Prob UP: {prob:.2f} | Pred: {pred_direction:8} | Actual: {actual_direction:8} | {'✅' if correct else '❌'}")

    # Summary
    results_df   = pd.DataFrame(results)
    total        = len(results_df)
    correct      = results_df["correct"].sum()
    accuracy     = (correct / total) * 100

    up_rows      = results_df[results_df["actual_direction"] == "UP"]
    down_rows    = results_df[results_df["actual_direction"] == "DOWN"]
    up_correct   = up_rows["correct"].mean() * 100 if len(up_rows) > 0 else 0
    down_correct = down_rows["correct"].mean() * 100 if len(down_rows) > 0 else 0

    print("\n" + "=" * 50)
    print(f"📊 BACKTEST RESULTS — {BACKTEST_TICKER}")
    print("=" * 50)
    print(f"  Total days tested  : {total}")
    print(f"  Correct direction  : {correct}/{total} ({accuracy:.1f}%)")
    print(f"  UP accuracy        : {up_correct:.1f}%")
    print(f"  DOWN accuracy      : {down_correct:.1f}%")
    print(f"  UP days in test    : {len(up_rows)}")
    print(f"  DOWN days in test  : {len(down_rows)}")
    print("=" * 50)

    results_df.to_csv(f"backtest_{BACKTEST_TICKER.replace('.', '_')}.csv", index=False)
    print(f"\n✅ Saved to backtest_{BACKTEST_TICKER.replace('.', '_')}.csv")

    return results_df

run_backtest()