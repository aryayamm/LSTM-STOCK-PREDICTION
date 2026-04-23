import requests
from datetime import datetime

def get_fear_greed():
    print("Fetching Fear & Greed index...")
    try:
        r    = requests.get("https://api.alternative.me/fng/?limit=10", timeout=10)
        data = r.json()["data"]

        # Today's value
        today     = data[0]
        value     = int(today["value"])
        label     = today["value_classification"]
        timestamp = today["timestamp"]

        # Last 5 days average
        recent    = [int(d["value"]) for d in data[:5]]
        avg_5d    = sum(recent) / len(recent)

        # Trend — is fear increasing or decreasing
        trend = "improving" if value > avg_5d else "worsening"

        print(f"  ✅ Fear & Greed: {value} ({label})")

        return {
            "value"    : value,
            "label"    : label,
            "avg_5d"   : avg_5d,
            "trend"    : trend,
            "normalized": value / 100  # 0-1 scale for model
        }

    except Exception as e:
        print(f"  ❌ Fear & Greed failed: {e}")
        return {
            "value"    : 50,
            "label"    : "Neutral",
            "avg_5d"   : 50,
            "trend"    : "neutral",
            "normalized": 0.5
        }

def get_fear_greed_history(days=90):
    """Get historical fear & greed for backtest"""
    try:
        r    = requests.get(f"https://api.alternative.me/fng/?limit={days}", timeout=10)
        data = r.json()["data"]

        history = {}
        for d in data:
            date  = datetime.fromtimestamp(int(d["timestamp"])).strftime("%Y-%m-%d")
            history[date] = int(d["value"]) / 100  # normalized 0-1

        return history
    except Exception as e:
        print(f"  ❌ Fear & Greed history failed: {e}")
        return {}