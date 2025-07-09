import streamlit as st
import numpy as np
import pickle
import shap
import json

# Load model
with open("sepsis_model_xgb.pkl", "rb") as f:
    model = pickle.load(f)

# Load feature names
with open("feature_names.json", "r") as f:
    feature_names = json.load(f)

st.set_page_config(page_title="Sepsis Risk Forecaster", layout="wide")

st.title("🧠 Sepsis Risk Prediction from ICU Vitals")
st.markdown("Enter patient vitals below to predict sepsis risk. If risk > 0.7, an alert will be shown.")

# Input form
with st.form("icu_form"):
    cols = st.columns(3)
    inputs = []
    for i, name in enumerate(feature_names):
        value = cols[i % 3].number_input(name, value=0.0, format="%.2f")
        inputs.append(value)
    submitted = st.form_submit_button("Predict Risk")

if submitted:
    X = np.array(inputs).reshape(1, -1)
    pred = model.predict_proba(X)[0][1]
    
    st.metric("🔍 Predicted Sepsis Risk", f"{pred:.2f}")

    if pred > 0.7:
        st.error("🚨 High risk of sepsis! Immediate attention needed.")
    else:
        st.success("✅ Sepsis risk under control.")

    # SHAP Explainability
    explainer = shap.Explainer(model)
    shap_values = explainer(X)
    st.subheader("🔬 SHAP Feature Importance")
    st.pyplot(shap.plots.waterfall(shap_values[0], max_display=10))
