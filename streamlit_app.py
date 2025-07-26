 import streamlit as st
 import pandas as pd
 import plotly.express as px
 from sklearn.ensemble import RandomForestRegressor
 from sklearn.model_selection import train_test_split
 from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
 
 # Sayfa ayarları
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
-
 def load_data():
-    return pd.read_csv("dataset_1.csv")
+    df = pd.read_csv("dataset.csv")
+    stats = df.describe().T
+    return df, stats
 
-df = load_data()
+df, dataset_stats = load_data()
 
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
diff --git a/streamlit_app.py b/streamlit_app.py
index f48336c65bf7a665efe04e696068bd08e7e5657f..f4e4c6f9ea381a559677f96a86e9dc7afc51f99f 100644
--- a/streamlit_app.py
+++ b/streamlit_app.py
@@ -61,40 +62,44 @@ r2 = r2_score(y_test, y_pred)
 
 st.subheader(f"Gerilim Düşümü Tahmin Performansı - {ev_sec}")
 col1, col2, col3 = st.columns(3)
 col1.metric("MAE", f"{mae:.2f} V")
 col2.metric("MSE", f"{mse:.2f}")
 col3.metric("R²", f"{r2:.3f}")
 
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
 c1, c2, c3 = st.columns(3)
 c1.metric("Doğruluk", f"{accuracy:.2f}")
 c2.metric("Kesinlik", f"{precision:.2f}")
 c3.metric("Duyarlılık", f"{recall:.2f}")
 
+# --- Dataset Summary Statistics ---
+st.subheader("Veri Kümesi İstatistikleri")
+st.dataframe(dataset_stats, use_container_width=True)
+
 # --- Grafik ---
 chart_df = pd.DataFrame({"Gerçek": y_test.values[:200], "Tahmin": y_pred[:200]})
 fig = px.line(chart_df, markers=True)
 fig.update_layout(
     title="Gerilim Düşümü Tahmini vs Gerçek Değer",
     yaxis_title="Gerilim Düşümü (V)",
     xaxis_title="Index",
 )
 st.plotly_chart(fig, use_container_width=True)
 
 # --- Feature Importance ---
 st.subheader("Özellik Önemleri")
 importances = model.feature_importances_
 importance_df = pd.DataFrame({"Feature": features, "Importance": importances}).sort_values(by="Importance", ascending=False)
 st.dataframe(importance_df, use_container_width=True)
