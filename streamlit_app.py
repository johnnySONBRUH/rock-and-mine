import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load model and data
model = joblib.load('prediction.pkl')
# Read without header
df = pd.read_csv('data.csv', header=None)

st.set_page_config(page_title="Sonar Predictor", page_icon="📡", layout="wide")
st.title("📡 Sonar: Rock vs Mine Predictor")
st.write("Enter 60 sonar feature values to classify. You can also load samples.")

# Initialize session state for inputs and sample version to force re-render
if 'input_data' not in st.session_state:
    st.session_state.input_data = [0.0] * 60
if 'sample_version' not in st.session_state:
    st.session_state.sample_version = 0

# Sample loading function
def load_sample(label):
    st.session_state.input_data = df[df[60] == label].iloc[0, :-1].tolist()
    st.session_state.sample_version += 1
    st.rerun()

col1, col2 = st.columns(2)
with col1:
    if st.button("Load Sample Rock"):
        load_sample('R')
with col2:
    if st.button("Load Sample Mine"):
        load_sample('M')

# Expanders for 60 features
for i in range(0, 60, 20):
    with st.expander(f"Features {i+1} - {i+20}"):
        cols = st.columns(4)
        for j in range(20):
            idx = i + j
            with cols[j % 4]:
                # Use key that changes on sample load to force update
                st.session_state.input_data[idx] = st.number_input(
                    f"F{idx+1}",
                    value=float(st.session_state.input_data[idx]),
                    format="%.4f",
                    key=f"feat_{idx}_{st.session_state.sample_version}"
                )

# Line chart for visualization
st.line_chart(st.session_state.input_data)

# Predict button
if st.button("Predict", type="primary"):
    arr = np.array(st.session_state.input_data).reshape(1, -1)
    pred = model.predict(arr)[0]
    result = "Rock" if pred == 'R' else "Mine"
    st.subheader("Prediction Result:")
    st.write(f"The object is classified as: **{result}**")

st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 20px;">
    <p>Made by bruzz-bruzz</p>
</div>
""", unsafe_allow_html=True)
