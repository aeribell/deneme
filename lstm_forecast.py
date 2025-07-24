import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

SEQ_LENGTH = 10


def load_series(path: str) -> np.ndarray:
    """Load and sort the voltage drop series."""
    df = pd.read_csv(path, parse_dates=["Timestamp"])
    df.sort_values("Timestamp", inplace=True)
    return df["Gerilim Düşümü (V)"].values.reshape(-1, 1)


def create_dataset(series: np.ndarray, length: int) -> tuple[np.ndarray, np.ndarray]:
    """Convert a 1D series to supervised learning format."""
    X, y = [], []
    for i in range(len(series) - length):
        X.append(series[i : i + length])
        y.append(series[i + length])
    return np.array(X), np.array(y)


def build_model(length: int) -> Sequential:
    model = Sequential(
        [
            LSTM(50, input_shape=(length, 1)),
            Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mse")
    return model


def main() -> None:
    series = load_series("dataset.csv")
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(series)
    X, y = create_dataset(scaled, SEQ_LENGTH)
    model = build_model(SEQ_LENGTH)
    model.fit(X, y, epochs=5, batch_size=32, verbose=1)
    preds = model.predict(X, verbose=0)
    mse = np.mean((preds - y) ** 2)
    print(f"MSE: {mse:.4f}")


if __name__ == "__main__":
    main()
