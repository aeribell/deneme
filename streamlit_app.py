import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Sayfa ayarları
st.set_page_config(page_title="Smart Grid AI", layout="wide")
st.title("🔌 Smart Grid - Gerilim Tahmini (AI Model)")

@st.cache_data

def load_data():
    return pd.read_csv("dataset.csv")

df = load_data()

# --- Sidebar ---
st.sidebar.header("Filtreler")
evler = df["Ev Adı"].unique()
ev_sec = st.sidebar.selectbox("Ev Seç", evler)
df_ev = df[df["Ev Adı"] == ev_sec]

# --- Özellikler ve Hedef ---
features = [
    "Power Consumption (kW)", "Reactive Power (kVAR)", "Power Factor",
    "Solar Power (kW)", "Wind Power (kW)", "Grid Supply (kW)",
    "Temperature (°C)", "Humidity (%)", "Electricity Price (USD/kWh)",
    "Predicted Load (kW)", "Güç Tüketimi (kW)", "Toplam Hat Mesafesi (m)"
]
target = "Gerilim (V)"

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

st.subheader(f"Model Performansı - {ev_sec}")
st.metric("MAE", f"{mae:.2f} V")
st.metric("MSE", f"{mse:.2f}")
st.metric("R²", f"{r2:.3f}")

# --- Grafik ---
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(y_test.values[:200], label="Gerçek", marker='o')
ax.plot(y_pred[:200], label="Tahmin", marker='x')
ax.set_title("Gerilim Tahmini vs Gerçek Değer")
ax.set_ylabel("Gerilim (V)")
ax.legend()
st.pyplot(fig)

# --- Feature Importance ---
st.subheader("Özellik Önemleri")
importances = model.feature_importances_
importance_df = pd.DataFrame({"Feature": features, "Importance": importances}).sort_values(by="Importance", ascending=False)
st.dataframe(importance_df, use_container_width=True)
