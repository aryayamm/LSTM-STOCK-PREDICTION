import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()

def get_conn():
    return psycopg2.connect(os.environ["DATABASE_URL"])

def init_db():
    conn = get_conn()
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id                INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
            date              TEXT,
            ticker            TEXT,
            predicted_price   REAL,
            actual_price      REAL,
            direction         TEXT,
            actual_direction  TEXT,
            change_predicted  REAL,
            change_actual     REAL,
            correct           INTEGER,
            -- technical
            rsi               REAL,
            macd              REAL,
            ma7               REAL,
            ma30              REAL,
            -- fundamental
            eps               REAL,
            roe               REAL,
            roa               REAL,
            der               REAL,
            pbv               REAL,
            per               REAL,
            market_cap        REAL,
            dividend_yield    REAL,
            -- sentiment
            sentiment_score   REAL,
            local_sentiment   TEXT,
            macro_sentiment   TEXT,
            final_sentiment   TEXT,
            -- timestamps
            created_at        TIMESTAMPTZ DEFAULT NOW()
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS news (
            id         INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
            date       TEXT,
            ticker     TEXT,
            type       TEXT,
            headline   TEXT,
            sentiment  TEXT,
            created_at TIMESTAMPTZ DEFAULT NOW()
        )
    """)

    conn.commit()
    conn.close()
    print("  ✅ NeonDB ready!")