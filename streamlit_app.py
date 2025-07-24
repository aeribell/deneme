import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Sayfa ayarları
st.set_page_config(page_title="Smart Grid AI", layout="wide")
st.title("🔌 Smart Grid - Gerilim Düşümü Tespiti")

@st.cache_data

def load_data():
    return pd.read_csv("dataset_1.csv")

df = load_data()

# --- Sidebar ---
st.sidebar.header("Filtreler")
evler = df["Ev Adı"].unique()
ev_sec = st.sidebar.selectbox("Ev Seç", evler)
df_ev = df[df["Ev Adı"] == ev_sec]
drop_threshold = st.sidebar.slider(
    "Gerilim Düşümü Eşik (V)", 0.2, 1.3, 0.9, 0.01
)

# --- Özellikler ve Hedef ---
features = [
    "Power Consumption (kW)", "Reactive Power (kVAR)", "Power Factor",
    "Solar Power (kW)", "Wind Power (kW)", "Grid Supply (kW)",
    "Temperature (°C)", "Humidity (%)", "Electricity Price (USD/kWh)",
    "Predicted Load (kW)", "Güç Tüketimi (kW)", "Toplam Hat Mesafesi (m)"
]
target = "Gerilim Düşümü (V)"

# --- Eğitim ---
X = df_ev[features]
y = df_ev[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# --- Metrix ---
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.subheader(f"Gerilim Düşümü Tahmin Performansı - {ev_sec}")
st.metric("MAE", f"{mae:.2f} V")
st.metric("MSE", f"{mse:.2f}")
st.metric("R²", f"{r2:.3f}")

# --- Gerilim Düşümü Tespiti ---
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
st.metric("Doğruluk", f"{accuracy:.2f}")
st.metric("Kesinlik", f"{precision:.2f}")
st.metric("Duyarlılık", f"{recall:.2f}")

# --- Grafik ---
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(y_test.values[:200], label="Gerçek", marker='o')
ax.plot(y_pred[:200], label="Tahmin", marker='x')
ax.set_title("Gerilim Düşümü Tahmini vs Gerçek Değer")
ax.set_ylabel("Gerilim Düşümü (V)")
ax.legend()
st.pyplot(fig)

# --- Feature Importance ---
st.subheader("Özellik Önemleri")
importances = model.feature_importances_
importance_df = pd.DataFrame({"Feature": features, "Importance": importances}).sort_values(by="Importance", ascending=False)
st.dataframe(importance_df, use_container_width=True)
