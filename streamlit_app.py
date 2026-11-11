import streamlit as st
import joblib
import numpy as np
import pandas as pd
model = joblib.load('prediction.pkl')
df = pd.read_csv('data.csv')
features = df.drop(columns=['R']).columns.tolist()
st.set_page_config(page_title="Sonar Predictor", page_icon="📡")
st.title("📡 Sonar: Rock vs Mine Predictor")
st.write("Enter the 60 sonar feature values to classify as Rock or Mine.")
input_data = []
cols = st.columns(3) 
for i, feature in enumerate(features):
    with cols[i % 3]:
        val = st.number_input(f"Feature {i+1}", value=0.0, format="%.4f")
        input_data.append(val)
if st.button("Predict", type="primary"):
    arr = np.array(input_data).reshape(1, -1)
    prediction = model.predict(arr)[0]
    st.subheader("Prediction Result:")
    st.write(f"The object is classified as: **{prediction}**")
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 20px;">
    <p>Made by bruzz-bruzz</p>
    <div style="display: flex; justify-content: center; gap: 20px;">
        <a href="YOUR_GITHUB_PROFILE_URL" target="_blank">GitHub Profile</a>
        <a href="YOUR_GITHUB_REPO_URL" target="_blank">GitHub Repository</a>
    </div>
</div>
""", unsafe_allow_html=True)