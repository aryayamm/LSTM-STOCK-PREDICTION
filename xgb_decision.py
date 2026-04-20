import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight

XGB_FEATURES = [
    # Technical
    "RSI", "MACD", "Signal", "MACD_Hist",
    "BB_width", "Stoch_K", "Stoch_D",
    "ROC5", "ROC10",
    "Volume_Ratio",
    "Price_MA7_dist", "Price_MA30_dist",
    # Fundamental
    "EPS", "ROE", "ROA", "PBV", "PER", "DividendYield",
    # Sentiment
    "Sentiment",
    # Sector
    "IHSG_Return",
    # LSTM signals
    "lstm_pred_price", "lstm_price_change", "lstm_momentum", "lstm_trend",
]

def create_labels(df):
    """
    BUY      → next day up > 0.5%
    SELL     → next day down > 0.5%
    NO TRADE → everything else
    """
    labels = []
    for i in range(len(df) - 1):
        next_return = (df["Close"].iloc[i + 1] - df["Close"].iloc[i]) / df["Close"].iloc[i] * 100
        if next_return > 0.5:
            labels.append("BUY")
        elif next_return < -0.5:
            labels.append("SELL")
        else:
            labels.append("NO_TRADE")
    labels.append("NO_TRADE")  # last row has no next day
    return labels

def train_xgboost(df):
    labels = create_labels(df)
    df     = df.copy()
    df["label"] = labels

    # Drop rows without LSTM signals
    df = df.dropna(subset=XGB_FEATURES)

    X = df[XGB_FEATURES].values
    y = df["label"].values

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Balance classes
    sample_weights = compute_sample_weight('balanced', y_encoded)

    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric='mlogloss',
        random_state=42,
        verbosity=0
    )

    split = int(len(X) * 0.8)
    model.fit(
        X[:split], y_encoded[:split],
        sample_weight=sample_weights[:split],
        eval_set=[(X[split:], y_encoded[split:])],
        verbose=False
    )

    return model, le

def get_xgb_decision(df_row, model, le):
    """Get BUY/SELL/NO_TRADE decision with confidence"""
    X = np.array([[df_row[f] for f in XGB_FEATURES]])
    probs    = model.predict_proba(X)[0]
    pred_idx = np.argmax(probs)
    decision = le.inverse_transform([pred_idx])[0]
    confidence = probs[pred_idx]

    # Map probs correctly using le.classes_
    prob_dict = {}
    for i, cls in enumerate(le.classes_):
        prob_dict[cls] = float(probs[i])

    return decision, confidence, prob_dict