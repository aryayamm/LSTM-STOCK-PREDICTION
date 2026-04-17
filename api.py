from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from database import get_conn
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Stock Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Latest prediction ──────────────────────────────
@app.get("/latest/{ticker}")
def get_latest(ticker: str):
    conn = get_conn()
    c    = conn.cursor()

    c.execute("""
        SELECT 
            date, ticker, predicted_price, actual_price,
            direction, actual_direction, change_predicted, change_actual, correct,
            rsi, macd, ma7, ma30,
            eps, roe, roa, der, pbv, per, market_cap, dividend_yield,
            sentiment_score, local_sentiment, macro_sentiment, final_sentiment,
            created_at
        FROM predictions
        WHERE ticker = %s
        ORDER BY date DESC LIMIT 1
    """, (ticker.upper() + ".JK" if ".JK" not in ticker.upper() else ticker.upper(),))

    row = c.fetchone()
    conn.close()

    if not row:
        raise HTTPException(status_code=404, detail=f"No prediction found for {ticker}")

    return {
        "date"            : row[0],
        "ticker"          : row[1],
        "predicted_price" : row[2],
        "actual_price"    : row[3],
        "direction"       : row[4],
        "actual_direction": row[5],
        "change_predicted": row[6],
        "change_actual"   : row[7],
        "correct"         : row[8],
        "technical": {
            "rsi" : row[9],
            "macd": row[10],
            "ma7" : row[11],
            "ma30": row[12],
        },
        "fundamental": {
            "eps"           : row[13],
            "roe"           : row[14],
            "roa"           : row[15],
            "der"           : row[16],
            "pbv"           : row[17],
            "per"           : row[18],
            "market_cap"    : row[19],
            "dividend_yield": row[20],
        },
        "sentiment": {
            "score"         : row[21],
            "local_label"   : row[22],
            "macro_label"   : row[23],
            "final_label"   : row[24],
        },
        "created_at": str(row[25])
    }

# ── History ────────────────────────────────────────
@app.get("/history/{ticker}")
def get_history(ticker: str, limit: int = 7):
    conn = get_conn()
    c    = conn.cursor()

    ticker = ticker.upper() + ".JK" if ".JK" not in ticker.upper() else ticker.upper()

    c.execute("""
        SELECT date, predicted_price, actual_price, direction, actual_direction, correct, change_predicted, change_actual
        FROM predictions
        WHERE ticker = %s
        ORDER BY date DESC
        LIMIT %s
    """, (ticker, limit))

    rows = c.fetchall()
    conn.close()

    return {
        "ticker": ticker,
        "history": [
            {
                "date"            : r[0],
                "predicted_price" : r[1],
                "actual_price"    : r[2],
                "direction"       : r[3],
                "actual_direction": r[4],
                "correct"         : r[5],
                "change_predicted": r[6],
                "change_actual"   : r[7],
            }
            for r in rows
        ]
    }

# ── Accuracy ───────────────────────────────────────
@app.get("/accuracy/{ticker}")
def get_accuracy(ticker: str):
    conn = get_conn()
    c    = conn.cursor()

    ticker = ticker.upper() + ".JK" if ".JK" not in ticker.upper() else ticker.upper()

    c.execute("""
        SELECT
            COUNT(*)     as total,
            SUM(correct) as correct,
            AVG(ABS(change_actual - change_predicted)) as avg_error
        FROM predictions
        WHERE ticker = %s AND correct IS NOT NULL
    """, (ticker,))

    row = c.fetchone()
    conn.close()

    if not row or row[0] == 0:
        return {"ticker": ticker, "message": "Not enough data yet"}

    total    = row[0]
    correct  = row[1] or 0
    avg_err  = row[2] or 0
    accuracy = (correct / total) * 100

    return {
        "ticker"   : ticker,
        "total"    : total,
        "correct"  : correct,
        "accuracy" : round(accuracy, 2),
        "avg_error": round(avg_err, 2),
    }

# ── News ───────────────────────────────────────────
@app.get("/news/{ticker}")
def get_news(ticker: str, limit: int = 10):
    conn = get_conn()
    c    = conn.cursor()

    ticker = ticker.upper() + ".JK" if ".JK" not in ticker.upper() else ticker.upper()

    c.execute("""
        SELECT date, type, headline, sentiment, created_at
        FROM news
        WHERE ticker = %s
        ORDER BY created_at DESC
        LIMIT %s
    """, (ticker, limit))

    rows = c.fetchall()
    conn.close()

    return {
        "ticker": ticker,
        "news": [
            {
                "date"     : r[0],
                "type"     : r[1],
                "headline" : r[2],
                "sentiment": r[3],
                "created_at": str(r[4])
            }
            for r in rows
        ]
    }

# ── Health check ───────────────────────────────────
@app.get("/")
def root():
    return {"status": "ok", "message": "Stock Prediction API is running 🚀"}