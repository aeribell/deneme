# Smart Grid Voltage Drop Detection

This repository contains a Streamlit application that trains a machine learning model to predict and detect voltage drop events for different houses in a smart grid.

## Running the Application

1. Install the dependencies:
   ```bash
   pip install pandas scikit-learn matplotlib plotly streamlit
   ```

2. Start the Streamlit app:
   ```bash
   streamlit run streamlit_app.py
   ```

Use the sidebar to select a house and adjust the voltage drop threshold. The app displays model metrics, voltage drop detection accuracy and feature importances.

## Dataset

The application loads `dataset.csv` at startup and computes basic summary statistics (mean, standard deviation, min and max) for all numeric columns. These statistics are displayed below the model metrics so you can better understand the ranges the model uses when detecting voltage drops.

## LSTM (Long Short-Term Memory) – Zaman Serisi Tahmini

Gerilim düşümünü geçmiş verilere bakarak zaman serisi şeklinde tahmin etmek için LSTM tabanlı bir model kullanılabilir. `lstm_forecast.py` dosyası, `dataset.csv` üzerinde temel bir LSTM eğitimi örneği içerir.
