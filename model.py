from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import numpy as np
from config import LOOK_BACK, EPOCHS, BATCH_SIZE

FEATURE_COLS = [
    "Close", "Volume", "MA7", "MA30", "RSI", "MACD", "Signal", "MACD_Hist",
    "BB_upper", "BB_lower", "BB_width",
    "ROC5", "ROC10",
    "Volume_Ratio",
    "Stoch_K", "Stoch_D",
    "Price_MA7_dist", "Price_MA30_dist",
    "Daily_Return",
    "EPS", "ROE", "ROA", "DER", "PBV", "PER", "MarketCap", "DividendYield",
    "Sentiment",
    "IHSG_Return", "IHSG_MA7", "IHSG_MA30",
]

def prepare_data(df):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[FEATURE_COLS])

    X, y = [], []
    for i in range(LOOK_BACK, len(scaled) - 1):
        X.append(scaled[i - LOOK_BACK:i])
        # 1 = UP, 0 = DOWN
        next_close = df["Close"].iloc[i + 1]
        curr_close = df["Close"].iloc[i]
        y.append(1 if next_close > curr_close else 0)

    X, y = np.array(X), np.array(y)

    # Balance classes — equal UP and DOWN samples
    from collections import Counter
    print(f"  Class balance: {Counter(y)}")

    split = int(len(X) * 0.8)
    return X[:split], X[split:], y[:split], y[split:], scaler

def build_model(input_shape):
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(1, activation="sigmoid")  # ← sigmoid for binary classification
    ])
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",  # ← classification loss
        metrics=["accuracy"]
    )
    return model

def train_and_predict(df):
    X_train, X_test, y_train, y_test, scaler = prepare_data(df)

    model = build_model((X_train.shape[1], X_train.shape[2]))

    from tensorflow.keras.callbacks import EarlyStopping
    from sklearn.utils.class_weight import compute_class_weight
    import numpy as np

    # Give DOWN moves more weight during training
    classes = np.unique(y_train)
    weights = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weight = dict(zip(classes, weights))
    print(f"  Class weights: {class_weight}")

    model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.1,
        class_weight=class_weight,  # ← force model to learn DOWN
        callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)],
        verbose=1
    )

    last_60 = scaler.transform(df[FEATURE_COLS].values[-LOOK_BACK:])
    last_60  = np.expand_dims(last_60, axis=0)

    prob = model.predict(last_60, verbose=0)[0][0]
    current_price = df["Close"].iloc[-1]

    # Convert probability to price
    # UP → current * 1.01, DOWN → current * 0.99
    if prob > 0.5:
        predicted_price = current_price * 1.01
    else:
        predicted_price = current_price * 0.99

    return current_price, predicted_price