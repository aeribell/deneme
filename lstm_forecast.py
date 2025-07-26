import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

DATA_FILE = "dataset.csv"
SEQUENCE_LENGTH = 24


def load_series():
    df = pd.read_csv(DATA_FILE, parse_dates=["Timestamp"])
    df.sort_values("Timestamp", inplace=True)

    features = [
        "Voltage (V)",
        "Current (A)",
        "Power Consumption (kW)",
        "Reactive Power (kVAR)"
    ]
    target = "Gerilim Düşümü (V)"

    data = df[features + [target]].astype(float)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data)

    X, y = [], []
    for i in range(SEQUENCE_LENGTH, len(scaled)):
        X.append(scaled[i - SEQUENCE_LENGTH:i, :-1])
        y.append(scaled[i, -1])

    return np.array(X), np.array(y)


def build_model(input_shape):
    model = Sequential([
        LSTM(64, input_shape=input_shape),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model


if __name__ == "__main__":
    X, y = load_series()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    model = build_model((SEQUENCE_LENGTH, X.shape[2]))
    model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.2)

    loss = model.evaluate(X_test, y_test)
    print(f"Test Loss: {loss:.4f}")
