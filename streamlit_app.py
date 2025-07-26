import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

st.set_page_config(page_title="Smart Grid AI", layout="wide")
st.markdown(
    """
    <style>
    .main {
        background-color: #f0f2f6;
        font-family: Arial, sans-serif;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
st.title("🔌 Smart Grid - Gerilim Düşümü Tespiti")

@st.cache_data
def load_data():
    df = pd.read_csv("dataset.csv")
    return df, df.describe().T

@st.cache_data
def load_series(seq_len: int = 24):
    df = pd.read_csv("dataset.csv", parse_dates=["Timestamp"])
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
    for i in range(seq_len, len(scaled)):
        X.append(scaled[i - seq_len:i, :-1])
        y.append(scaled[i, -1])
    return np.array(X), np.array(y)

def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(64, input_shape=input_shape),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model

def random_forest_page(df: pd.DataFrame, stats: pd.DataFrame):
    st.sidebar.header("Filtreler")
    evler = df["Ev Adı"].unique()
    ev_sec = st.sidebar.selectbox("Ev Seç", evler)
    df_ev = df[df["Ev Adı"] == ev_sec]
    drop_threshold = st.sidebar.slider(
        "Gerilim Düşümü Eşik (V)", 0.2, 1.3, 0.9, 0.01
    )

    features = [
        "Power Consumption (kW)", "Reactive Power (kVAR)", "Power Factor",
        "Solar Power (kW)", "Wind Power (kW)", "Grid Supply (kW)",
        "Temperature (°C)", "Humidity (%)", "Electricity Price (USD/kWh)",
        "Güç Tüketimi (kW)", "Toplam Hat Mesafesi (m)"
    ]
    target = "Gerilim Düşümü (V)"

    X = df_ev[features]
    y = df_ev[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    st.subheader(f"Gerilim Düşümü Tahmin Performansı - {ev_sec}")
    col1, col2, col3 = st.columns(3)
    col1.metric("MAE", f"{mae:.2f} V")
    col2.metric("MSE", f"{mse:.2f}")
    col3.metric("R²", f"{r2:.3f}")

    y_pred_series = pd.Series(y_pred, index=y_test.index)
    y_test_drop = y_test >= drop_threshold
    y_pred_drop = y_pred_series >= drop_threshold
    tp = ((y_pred_drop) & (y_test_drop)).sum()
    tn = ((~y_pred_drop) & (~y_test_drop)).sum()
    fp = ((y_pred_drop) & (~y_test_drop)).sum()
    fn = ((~y_pred_drop) & (y_test_drop)).sum()
    accuracy = (tp + tn) / len(y_test_drop)
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0

    st.subheader("Gerilim Düşümü Tespit Metriği")
    c1, c2, c3 = st.columns(3)
    c1.metric("Doğruluk", f"{accuracy:.2f}")
    c2.metric("Kesinlik", f"{precision:.2f}")
    c3.metric("Duyarlılık", f"{recall:.2f}")

    st.subheader("Veri Kümesi İstatistikleri")
    st.dataframe(stats, use_container_width=True)

    chart_df = pd.DataFrame({"Gerçek": y_test.values[:200], "Tahmin": y_pred[:200]})
    fig = px.line(chart_df, markers=True)
    fig.update_layout(
        title="Gerilim Düşümü Tahmini vs Gerçek Değer",
        yaxis_title="Gerilim Düşümü (V)",
        xaxis_title="Index",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Özellik Önemleri")
    importances = model.feature_importances_
    importance_df = pd.DataFrame({"Feature": features, "Importance": importances}).sort_values(
        by="Importance", ascending=False
    )
    st.dataframe(importance_df, use_container_width=True)

def lstm_page():
    st.header("LSTM Zaman Serisi Tahmini")
    if st.button("Modeli Eğit"):
        with st.spinner("Eğitiliyor..."):
            X, y = load_series()
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, shuffle=False
            )
            model = build_lstm_model((X.shape[1], X.shape[2]))
            model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.2)
            loss = model.evaluate(X_test, y_test)
        st.success(f"Test Loss: {loss:.4f}")
    else:
        st.write("Eğitimi başlatmak için 'Modeli Eğit' düğmesine tıklayın.")

df, dataset_stats = load_data()

page = st.sidebar.selectbox("Sayfa", ["Random Forest", "LSTM Tahmini"])

if page == "Random Forest":
    random_forest_page(df, dataset_stats)
else:
    lstm_page()
