import requests
import numpy as np
import xml.etree.ElementTree as ET
from transformers import pipeline
from deep_translator import GoogleTranslator
from datetime import datetime, timedelta
import email.utils
from config import TICKER_NAME

# ── Macro keywords that affect Indonesian banking stocks ──
MACRO_QUERIES = [
    "https://news.google.com/rss/search?q=Federal+Reserve+interest+rate&hl=en-US&gl=US&ceid=US:en",
    "https://news.google.com/rss/search?q=USD+IDR+rupiah+exchange+rate&hl=en-US&gl=US&ceid=US:en",
    "https://news.google.com/rss/search?q=oil+price+OPEC&hl=en-US&gl=US&ceid=US:en",
    "https://news.google.com/rss/search?q=US+China+trade+war+tariff&hl=en-US&gl=US&ceid=US:en",
    "https://news.google.com/rss/search?q=geopolitical+risk+emerging+markets&hl=en-US&gl=US&ceid=US:en",
]

# ── Local BBRI news ──
LOCAL_QUERIES = [
    f"https://news.google.com/rss/search?q={TICKER_NAME}&hl=en-US&gl=US&ceid=US:en",
    f"https://news.google.com/rss/search?q=BBRI+saham+indonesia&hl=id&gl=ID&ceid=ID:id",
    f"https://news.google.com/rss/search?q=Bank+Rakyat+Indonesia+saham&hl=id&gl=ID&ceid=ID:id",
]

def translate_to_english(text):
    try:
        return GoogleTranslator(source='auto', target='en').translate(text)
    except:
        return text

def fetch_headlines(queries, label=""):
    headlines = []
    for url in queries:
        try:
            r    = requests.get(url, timeout=10)
            root = ET.fromstring(r.content)
            items = root.findall(".//item")
            print(f"  Found {len(items)} articles [{label}]")

            for item in items[:10]:
                title    = item.find("title")
                pub_date = item.find("pubDate")

                if title is None:
                    continue

                # Filter last 1 week
                if pub_date is not None:
                    try:
                        date = email.utils.parsedate_to_datetime(pub_date.text)
                        age  = datetime.now(date.tzinfo) - date
                        if age > timedelta(weeks=1):
                            continue
                    except:
                        pass

                text       = title.text
                translated = translate_to_english(text)
                headlines.append(translated)

        except Exception as e:
            print(f"  ❌ Failed: {e}")

    return headlines

def analyze_sentiment(headlines, label=""):
    if not headlines:
        print(f"  ⚠️ No headlines for {label}, using neutral")
        return 0.5, []

    print(f"\n  Analyzing {len(headlines)} {label} headlines with FinBERT...")

    sentiment_pipeline = pipeline(
        "sentiment-analysis",
        model="ProsusAI/finbert",
        tokenizer="ProsusAI/finbert"
    )

    scores  = []
    results = []
    for headline in headlines:
        try:
            result = sentiment_pipeline(headline[:512])[0]
            label_  = result["label"]
            score   = result["score"]

            if label_ == "positive":
                scores.append(score)
                results.append(f"🟢 {headline[:60]}...")
            elif label_ == "negative":
                scores.append(-score)
                results.append(f"🔴 {headline[:60]}...")
            else:
                scores.append(0)
                results.append(f"🟡 {headline[:60]}...")
        except:
            pass

    avg_score  = np.mean(scores) if scores else 0
    normalized = (avg_score + 1) / 2
    return normalized, results

def get_news_sentiment():
    print("Fetching local BBRI news...")
    local_headlines = fetch_headlines(LOCAL_QUERIES, label="LOCAL")
    local_score, local_results = analyze_sentiment(local_headlines, label="local")

    print("\nFetching macro news...")
    macro_headlines = fetch_headlines(MACRO_QUERIES, label="MACRO")
    macro_score, macro_results = analyze_sentiment(macro_headlines, label="macro")

    # Weighted combination — local matters more
    final_score = (local_score * 0.7) + (macro_score * 0.3)

    def score_to_label(score):
        avg = (score * 2) - 1  # back to -1 to 1
        if avg > 0.1:
            return "Positive 🟢"
        elif avg < -0.1:
            return "Negative 🔴"
        else:
            return "Neutral 🟡"

    local_label = score_to_label(local_score)
    macro_label = score_to_label(macro_score)
    final_label = score_to_label(final_score)

    return final_score, local_results, macro_results, local_label, macro_label, final_label

def add_sentiment(df, sentiment_score):
    df["Sentiment"] = sentiment_score
    return df