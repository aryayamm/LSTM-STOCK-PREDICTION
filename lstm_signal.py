import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from config import LOOK_BACK, EPOCHS, BATCH_SIZE
import tensorflow as tf
import numpy as np

tf.random.set_seed(42)
np.random.seed(42)

LSTM_FEATURES = [
    "Close", "Volume", "MA7", "MA30",
    "BB_upper", "BB_lower", "BB_width",
    "ROC5", "ROC10",
    "Daily_Return",
    "IHSG_Return", "IHSG_MA7", "IHSG_MA30",
]

def build_lstm(input_shape):
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape),
        BatchNormalization(),
        Dropout(0.2),
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(1)  # predicts next price
    ])
    model.compile(optimizer="adam", loss="huber")
    return model

def train_lstm(df):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[LSTM_FEATURES])

    close_scaler = MinMaxScaler()
    close_scaler.fit_transform(df[["Close"]])

    X, y = [], []
    for i in range(LOOK_BACK, len(scaled)):
        X.append(scaled[i - LOOK_BACK:i])
        y.append(scaled[i][0])

    X, y = np.array(X), np.array(y)
    split = int(len(X) * 0.8)

    model = build_lstm((X.shape[1], X.shape[2]))
    model.fit(
        X[:split], y[:split],
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.1,
        callbacks=[EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)],
        verbose=1
    )

    return model, scaler, close_scaler

def get_lstm_signals(df, model, scaler, close_scaler):
    """Generate LSTM signals for every row in df"""
    scaled = scaler.transform(df[LSTM_FEATURES])
    signals = []

    for i in range(LOOK_BACK, len(scaled)):
        window = scaled[i - LOOK_BACK:i]
        window = np.expand_dims(window, axis=0)

        pred_scaled   = model.predict(window, verbose=0)[0][0]
        pred_price    = close_scaler.inverse_transform([[pred_scaled]])[0][0]
        current_price = df["Close"].iloc[i]
        prev_price    = df["Close"].iloc[i - 1]

        # LSTM signals
        price_change     = (pred_price - current_price) / current_price * 100
        momentum         = (current_price - prev_price) / prev_price * 100
        trend            = 1 if df["MA7"].iloc[i] > df["MA30"].iloc[i] else -1

        signals.append({
            "lstm_pred_price"  : pred_price,
            "lstm_price_change": price_change,
            "lstm_momentum"    : momentum,
            "lstm_trend"       : trend,
        })

    return signals