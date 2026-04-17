import sqlite3
import os
from datetime import datetime

DB_PATH = "predictions.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            date            TEXT,
            ticker          TEXT,
            predicted_price REAL,
            actual_price    REAL,
            direction       TEXT,
            actual_direction TEXT,
            change_predicted REAL,
            change_actual    REAL,
            correct         INTEGER
        )
    """)
    conn.commit()
    conn.close()
    print("  ✅ Database ready!")

def save_prediction(ticker, predicted_price, current_price, direction, change):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    today = datetime.now().strftime("%Y-%m-%d")

    # Check if prediction for today already exists
    c.execute("SELECT id FROM predictions WHERE date = ? AND ticker = ?", (today, ticker))
    existing = c.fetchone()

    if existing:
        # Update instead of insert
        c.execute("""
            UPDATE predictions
            SET predicted_price = ?, direction = ?, change_predicted = ?
            WHERE date = ? AND ticker = ?
        """, (predicted_price, direction, change, today, ticker))
        print("  ✅ Prediction updated for today!")
    else:
        c.execute("""
            INSERT INTO predictions (date, ticker, predicted_price, direction, change_predicted)
            VALUES (?, ?, ?, ?, ?)
        """, (today, ticker, predicted_price, direction, change))
        print("  ✅ Prediction saved!")

    conn.commit()
    conn.close()

def update_actual(ticker, actual_price):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # Find yesterday's prediction
    c.execute("""
        SELECT id, predicted_price, direction
        FROM predictions
        WHERE ticker = ? AND actual_price IS NULL
        ORDER BY date DESC LIMIT 1
    """, (ticker,))
    row = c.fetchone()

    if not row:
        print("  ⚠️ No pending prediction to update")
        conn.close()
        return None

    pred_id        = row[0]
    predicted_price = row[1]
    pred_direction  = row[2]

    # Calculate actual change
    c.execute("SELECT predicted_price FROM predictions WHERE id = ?", (pred_id,))
    prev = c.fetchone()

    change_actual    = ((actual_price - predicted_price) / predicted_price) * 100
    actual_direction = "🟢 UP" if change_actual > 0.1 else "🔴 DOWN" if change_actual < -0.1 else "🟡 SIDEWAYS"
    correct          = 1 if actual_direction == pred_direction else 0

    c.execute("""
        UPDATE predictions
        SET actual_price = ?, actual_direction = ?, change_actual = ?, correct = ?
        WHERE id = ?
    """, (actual_price, actual_direction, change_actual, correct, pred_id))

    conn.commit()
    conn.close()

    print(f"  ✅ Actual price updated!")
    return correct

def get_accuracy_summary(ticker):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute("""
        SELECT 
            COUNT(*) as total,
            SUM(correct) as correct,
            AVG(ABS(change_actual - change_predicted)) as avg_error
        FROM predictions
        WHERE ticker = ? AND correct IS NOT NULL
    """, (ticker,))
    row = c.fetchone()
    conn.close()

    if not row or row[0] == 0:
        return None

    total   = row[0]
    correct = row[1] or 0
    avg_err = row[2] or 0
    accuracy = (correct / total) * 100

    return {
        "total"    : total,
        "correct"  : correct,
        "accuracy" : accuracy,
        "avg_error": avg_err
    }

def get_history(ticker, limit=7):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        SELECT date, predicted_price, actual_price, direction, actual_direction, correct
        FROM predictions
        WHERE ticker = ?
        ORDER BY date DESC
        LIMIT ?
    """, (ticker, limit))
    rows = c.fetchall()
    conn.close()
    return rows