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
            rsi               REAL,
            macd              REAL,
            ma7               REAL,
            ma30              REAL,
            eps               REAL,
            roe               REAL,
            roa               REAL,
            der               REAL,
            pbv               REAL,
            per               REAL,
            market_cap        REAL,
            dividend_yield    REAL,
            sentiment_score   REAL,
            local_sentiment   TEXT,
            macro_sentiment   TEXT,
            final_sentiment   TEXT,
            decision          TEXT,
            confidence        REAL,
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

    c.execute("""
        CREATE TABLE IF NOT EXISTS paper_trades (
            id            INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
            date          TEXT,
            ticker        TEXT,
            decision      TEXT,
            confidence    REAL,
            entry_price   REAL,
            exit_price    REAL,
            pnl_pct       REAL,
            pnl_rp        REAL,
            capital_after REAL,
            correct       INTEGER,
            created_at    TIMESTAMPTZ DEFAULT NOW()
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS paper_portfolio (
            id          INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
            ticker      TEXT UNIQUE,
            capital     REAL,
            updated_at  TIMESTAMPTZ DEFAULT NOW()
        )
    """)

    conn.commit()
    conn.close()
    print("  ✅ NeonDB ready!")