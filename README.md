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
