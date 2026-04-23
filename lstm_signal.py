import os
os.environ['PYTHONHASHSEED'] = '42'
os.environ['TF_DETERMINISTIC_OPS'] = '1'
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from config import LOOK_BACK, EPOCHS, BATCH_SIZE

tf.random.set_seed(42)
np.random.seed(42)

LSTM_FEATURES = [
    "Close", "Volume", "MA7", "MA30",
    "BB_upper", "BB_lower", "BB_width",
    "ROC5", "ROC10",
    "Daily_Return",
    "FearGreed",           # ← new
    "IHSG_Return",    "IHSG_MA7",    "IHSG_MA30",
    "Nikkei_Return",  "Nikkei_MA7",  "Nikkei_MA30",
    "KOSPI_Return",   "KOSPI_MA7",   "KOSPI_MA30",
    "HangSeng_Return","HangSeng_MA7","HangSeng_MA30",
    "SGX_Return",     "SGX_MA7",     "SGX_MA30",
]

def build_lstm(input_shape):
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape),
        BatchNormalization(),
        Dropout(0.2),
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="huber")
    return model

def train_lstm(df):
    split = int(len(df) * 0.8)
    train_df = df.iloc[:split]

    scaler = MinMaxScaler()
    scaler.fit(train_df[LSTM_FEATURES])
    scaled = scaler.transform(df[LSTM_FEATURES])

    # Predict RETURN not price
    returns = df["Close"].pct_change().fillna(0).values

    X, y = [], []
    for i in range(LOOK_BACK, split):
        X.append(scaled[i - LOOK_BACK:i])
        y.append(returns[i])  # ← % change, not price

    X, y = np.array(X), np.array(y)

    model = build_lstm((X.shape[1], X.shape[2]))
    val_split = int(len(X) * 0.9)
    model.fit(
        X[:val_split], y[:val_split],
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X[val_split:], y[val_split:]),
        callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)],
        verbose=1
    )

    return model, scaler

def get_lstm_signals(df, model, scaler):
    scaled  = scaler.transform(df[LSTM_FEATURES])
    signals = []

    for i in range(LOOK_BACK, len(scaled)):
        window = scaled[i - LOOK_BACK:i]
        window = np.expand_dims(window, axis=0)

        pred_return   = float(model.predict(window, verbose=0)[0][0])
        current_price = float(df["Close"].iloc[i])
        prev_price    = float(df["Close"].iloc[i - 1])

        # Cap return at ±3%
        pred_return   = max(min(pred_return, 0.03), -0.03)
        pred_price    = current_price * (1 + pred_return)
        price_change  = pred_return * 100
        momentum      = (current_price - prev_price) / prev_price * 100
        trend         = 1 if df["MA7"].iloc[i] > df["MA30"].iloc[i] else -1

        signals.append({
            "lstm_pred_price"  : pred_price,
            "lstm_price_change": price_change,
            "lstm_momentum"    : momentum,
            "lstm_trend"       : trend,
        })

    return signals