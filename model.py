import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from config import LOOK_BACK, EPOCHS, BATCH_SIZE

FEATURE_COLS = [
    "Close", "Volume", "MA7", "MA30", "RSI", "MACD", "Signal",
    "BB_upper", "BB_lower",
    "EPS", "ROE", "ROA", "DER", "PBV", "PER", "MarketCap", "DividendYield",
    "Sentiment",
    "IHSG_Return", "IHSG_MA7", "IHSG_MA30",
    "Banking_Return", "Banking_MA7", "Banking_MA30",
]

def prepare_data(df):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[FEATURE_COLS])

    close_scaler = MinMaxScaler()
    close_scaler.fit_transform(df[["Close"]])

    X, y = [], []
    for i in range(LOOK_BACK, len(scaled)):
        X.append(scaled[i - LOOK_BACK:i])
        y.append(scaled[i][0])

    X, y = np.array(X), np.array(y)
    split = int(len(X) * 0.8)
    return X[:split], X[split:], y[:split], y[split:], scaler, close_scaler

def build_model(input_shape):
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model

def train_and_predict(df):
    X_train, X_test, y_train, y_test, scaler, close_scaler = prepare_data(df)

    print("Training LSTM model...")
    model = build_model((X_train.shape[1], X_train.shape[2]))
    model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)

    last_60 = scaler.transform(df[FEATURE_COLS].values[-LOOK_BACK:])
    last_60 = np.expand_dims(last_60, axis=0)

    predicted_scaled = model.predict(last_60, verbose=0)[0][0]
    predicted_price = close_scaler.inverse_transform([[predicted_scaled]])[0][0]
    current_price = df["Close"].iloc[-1]

    return current_price, predicted_price