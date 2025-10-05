# app.py
import pickle
import numpy as np
import streamlit as st

st.title("Coral Cove Treasure Predictor")

PKL_PATH = "model.pkl"   # make sure this file is in the same folder

@st.cache_resource
def load_model():
    with open(PKL_PATH, "rb") as f:
        return pickle.load(f)

# load once (cached)
try:
    model = load_model()
    st.success(f"Loaded: {PKL_PATH}")
except Exception as e:
    st.error(f"Could not load '{PKL_PATH}': {e}")
    st.stop()

st.write("Enter features and click Predict:")

area = st.number_input("Area (10–1000)", 10, 1000, 250, 1)
fish = st.number_input("Fish Population (100–10000)", 100, 10000, 3000, 10)
seaweed = st.number_input("Seaweed Density (0–100)", 0, 100, 50, 1)

if st.button("Predict"):
    X = np.array([[area, fish, seaweed]], dtype=float)  # order must match training
    try:
        y_pred = model.predict(X)[0]
        st.subheader(f"Estimated Treasure Value: {int(round(y_pred))}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
