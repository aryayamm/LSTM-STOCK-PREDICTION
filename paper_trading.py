from database import get_conn
from datetime import datetime

INITIAL_CAPITAL = 10_000_000  # Rp 10 juta per ticker

def init_paper_trading():
    conn = get_conn()
    c    = conn.cursor()
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

def get_current_capital(ticker):
    conn = get_conn()
    c    = conn.cursor()
    c.execute("SELECT capital FROM paper_portfolio WHERE ticker = %s", (ticker,))
    row = c.fetchone()
    conn.close()
    if row:
        return row[0]
    return INITIAL_CAPITAL

def record_trade(ticker, result):
    init_paper_trading()

    decision      = result["decision"]
    confidence    = result["confidence"]
    current_price = result["current_price"]
    today         = datetime.now().strftime("%Y-%m-%d")

    if decision == "NO_TRADE":
        print(f"  ⏭ NO TRADE — skipping paper trade")
        return

    capital = get_current_capital(ticker)

    # Update yesterday's trade exit price
    conn = get_conn()
    c    = conn.cursor()

    c.execute("""
        SELECT id, decision, entry_price, capital_after
        FROM paper_trades
        WHERE ticker = %s AND exit_price IS NULL
        ORDER BY date DESC LIMIT 1
    """, (ticker,))
    pending = c.fetchone()

    if pending:
        trade_id   = pending[0]
        prev_dec   = pending[1]
        entry      = pending[2]
        prev_cap   = pending[3]

        pnl_pct = ((current_price - entry) / entry) * 100
        if prev_dec == "SELL":
            pnl_pct = -pnl_pct

        pnl_rp        = prev_cap * (pnl_pct / 100)
        capital_after = prev_cap + pnl_rp
        correct       = 1 if pnl_pct > 0 else 0

        c.execute("""
            UPDATE paper_trades
            SET exit_price = %s, pnl_pct = %s, pnl_rp = %s,
                capital_after = %s, correct = %s
            WHERE id = %s
        """, (current_price, pnl_pct, pnl_rp, capital_after, correct, trade_id))

        capital = capital_after
        print(f"  ✅ Previous trade closed: PnL {pnl_pct:+.2f}% (Rp {pnl_rp:+,.0f})")

    # Record new trade
    c.execute("""
        INSERT INTO paper_trades
        (date, ticker, decision, confidence, entry_price, capital_after)
        VALUES (%s, %s, %s, %s, %s, %s)
    """, (today, ticker, decision, confidence, current_price, capital))

    # Update portfolio capital
    c.execute("""
        INSERT INTO paper_portfolio (ticker, capital)
        VALUES (%s, %s)
        ON CONFLICT (ticker) DO UPDATE SET capital = %s, updated_at = NOW()
    """, (ticker, capital, capital))

    conn.commit()
    conn.close()
    print(f"  ✅ New {decision} trade recorded at Rp {current_price:,.0f}")

def get_portfolio_summary(ticker):
    init_paper_trading()
    conn = get_conn()
    c    = conn.cursor()

    capital = get_current_capital(ticker)

    c.execute("""
        SELECT
            COUNT(*)            as total_trades,
            SUM(CASE WHEN correct = 1 THEN 1 ELSE 0 END) as wins,
            SUM(pnl_rp)         as total_pnl_rp,
            AVG(pnl_pct)        as avg_pnl_pct
        FROM paper_trades
        WHERE ticker = %s AND exit_price IS NOT NULL
    """, (ticker,))
    row = c.fetchone()
    conn.close()

    if not row or row[0] == 0:
        return None

    total_trades = row[0]
    wins         = row[1] or 0
    total_pnl_rp = row[2] or 0
    avg_pnl_pct  = row[3] or 0
    win_rate     = (wins / total_trades) * 100 if total_trades > 0 else 0
    total_pnl    = ((capital - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100

    return {
        "capital"      : capital,
        "total_value"  : capital,
        "total_pnl"    : total_pnl,
        "total_pnl_rp" : total_pnl_rp,
        "total_trades" : total_trades,
        "wins"         : wins,
        "win_rate"     : win_rate,
        "avg_pnl_pct"  : avg_pnl_pct,
    }