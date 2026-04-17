from database import get_conn
from datetime import datetime

def to_float(val):
    try:
        return float(val)
    except:
        return 0.0

def save_prediction(ticker, result):
    conn = get_conn()
    c    = conn.cursor()
    today = datetime.now().strftime("%Y-%m-%d")

    t = result["technical"]
    f = result["fundamental"]
    s = result["sentiment"]

    c.execute("SELECT id FROM predictions WHERE date = %s AND ticker = %s", (today, ticker))
    existing = c.fetchone()

    if existing:
        c.execute("""
            UPDATE predictions SET
                predicted_price = %s, direction = %s, change_predicted = %s,
                rsi = %s, macd = %s, ma7 = %s, ma30 = %s,
                eps = %s, roe = %s, roa = %s, der = %s, pbv = %s, per = %s,
                market_cap = %s, dividend_yield = %s,
                sentiment_score = %s, local_sentiment = %s,
                macro_sentiment = %s, final_sentiment = %s
            WHERE date = %s AND ticker = %s
        """, (
            to_float(result["predicted_price"]), result["direction"], to_float(result["change"]),
            to_float(t["rsi"]), to_float(t["macd"]), to_float(t["ma7"]), to_float(t["ma30"]),
            to_float(f["EPS"]), to_float(f["ROE"]), to_float(f["ROA"]), to_float(f["DER"]),
            to_float(f["PBV"]), to_float(f["PER"]), to_float(f["MarketCap"]), to_float(f["DividendYield"]),
            to_float(s["score"]), s["local_label"], s["macro_label"], s["final_label"],
            today, ticker
        ))
        print("  ✅ Prediction updated!")
    else:
        c.execute("""
            INSERT INTO predictions (
                date, ticker, predicted_price, direction, change_predicted,
                rsi, macd, ma7, ma30,
                eps, roe, roa, der, pbv, per, market_cap, dividend_yield,
                sentiment_score, local_sentiment, macro_sentiment, final_sentiment
            ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        """, (
            today, ticker, to_float(result["predicted_price"]), result["direction"], to_float(result["change"]),
            to_float(t["rsi"]), to_float(t["macd"]), to_float(t["ma7"]), to_float(t["ma30"]),
            to_float(f["EPS"]), to_float(f["ROE"]), to_float(f["ROA"]), to_float(f["DER"]),
            to_float(f["PBV"]), to_float(f["PER"]), to_float(f["MarketCap"]), to_float(f["DividendYield"]),
            to_float(s["score"]), s["local_label"], s["macro_label"], s["final_label"]
        ))
        print("  ✅ Prediction saved!")

    for headline in s["local_news"]:
        c.execute("""
            INSERT INTO news (date, ticker, type, headline)
            VALUES (%s, %s, %s, %s)
        """, (today, ticker, "local", headline))

    for headline in s["macro_news"]:
        c.execute("""
            INSERT INTO news (date, ticker, type, headline)
            VALUES (%s, %s, %s, %s)
        """, (today, ticker, "macro", headline))

    conn.commit()
    conn.close()

def update_actual(ticker, actual_price):
    conn = get_conn()
    c    = conn.cursor()

    c.execute("""
        SELECT id, predicted_price, direction
        FROM predictions
        WHERE ticker = %s AND actual_price IS NULL
        ORDER BY date DESC LIMIT 1
    """, (ticker,))
    row = c.fetchone()

    if not row:
        print("  ⚠️ No pending prediction to update")
        conn.close()
        return None

    pred_id         = row[0]
    predicted_price = row[1]
    pred_direction  = row[2]

    change_actual    = ((actual_price - predicted_price) / predicted_price) * 100
    actual_direction = "UP" if change_actual > 0.1 else "DOWN" if change_actual < -0.1 else "SIDEWAYS"
    correct          = 1 if actual_direction == pred_direction else 0

    c.execute("""
        UPDATE predictions
        SET actual_price = %s, actual_direction = %s, change_actual = %s, correct = %s
        WHERE id = %s
    """, (actual_price, actual_direction, change_actual, correct, pred_id))

    conn.commit()
    conn.close()
    print("  ✅ Actual price updated!")
    return correct

def get_accuracy_summary(ticker):
    conn = get_conn()
    c    = conn.cursor()

    c.execute("""
        SELECT
            COUNT(*)        as total,
            SUM(correct)    as correct,
            AVG(ABS(change_actual - change_predicted)) as avg_error
        FROM predictions
        WHERE ticker = %s AND correct IS NOT NULL
    """, (ticker,))
    row = c.fetchone()
    conn.close()

    if not row or row[0] == 0:
        return None

    total    = row[0]
    correct  = row[1] or 0
    avg_err  = row[2] or 0
    accuracy = (correct / total) * 100

    return {
        "total"    : total,
        "correct"  : correct,
        "accuracy" : accuracy,
        "avg_error": avg_err
    }

def get_history(ticker, limit=7):
    conn = get_conn()
    c    = conn.cursor()

    c.execute("""
        SELECT date, predicted_price, actual_price, direction, actual_direction, correct
        FROM predictions
        WHERE ticker = %s
        ORDER BY date DESC
        LIMIT %s
    """, (ticker, limit))
    rows = c.fetchall()
    conn.close()
    return rows