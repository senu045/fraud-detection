
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Fraud Detection Engine", page_icon="💳", layout="wide")

@st.cache_resource
def load_artifacts():
    model         = joblib.load("fraud_model.pkl")
    feature_names = joblib.load("feature_names.pkl")
    scaler_stats  = joblib.load("scaler_stats.pkl")
    return model, feature_names, scaler_stats

model, feature_names, scaler_stats = load_artifacts()

st.title("💳 Credit Card Fraud Detection Engine")
st.caption("XGBoost + SMOTE + SHAP  |  Trained on 284,807 real transactions  |  PR-AUC: 0.877")
st.divider()

st.sidebar.title("🔧 Transaction Input")
amount    = st.sidebar.number_input("Transaction Amount ($)", 0.0, 25000.0, 150.0, step=10.0)
hour      = st.sidebar.slider("Hour of Day (0 = midnight)", 0, 23, 14)
threshold = st.sidebar.slider("Decision Threshold", 0.10, 0.90, 0.50, 0.05)
st.sidebar.caption("Lower threshold = catch more fraud. Higher = fewer false alarms.")
st.sidebar.divider()

with st.sidebar.expander("Edit PCA Features V1-V28 (Advanced)"):
    v_vals = {}
    for i in range(1, 29):
        v_vals[f"V{i}"] = st.number_input(f"V{i}", -10.0, 10.0, 0.0, step=0.1, key=f"v{i}")

input_dict = {f"V{i}": v_vals[f"V{i}"] for i in range(1, 29)}
input_dict["Amount_scaled"] = (amount - scaler_stats["amount_mean"]) / scaler_stats["amount_std"]
input_dict["Time_scaled"]   = ((hour * 3600) - scaler_stats["time_mean"]) / scaler_stats["time_std"]
input_df = pd.DataFrame([input_dict])[feature_names]

fraud_proba = model.predict_proba(input_df)[0][1]
is_fraud    = fraud_proba >= threshold
proba_pct   = round(float(fraud_proba) * 100, 1)
proba_float = float(fraud_proba)

st.subheader("📊 Prediction Result")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Fraud Probability", f"{proba_float:.1%}")
col2.metric("Decision Threshold", f"{threshold:.0%}")
col3.metric("Amount", f"${amount:,.2f}")
col4.metric("Hour of Day", f"{hour}:00")

if is_fraud:
    st.error(f"🚨 FRAUD DETECTED — This transaction would be BLOCKED  ({proba_pct}% fraud probability)")
else:
    st.success(f"✅ LEGITIMATE — This transaction would be APPROVED  ({proba_pct}% fraud probability)")

st.caption("Risk Score")
st.progress(proba_float)

st.divider()

col_left, col_right = st.columns([3, 2])

with col_left:
    st.subheader("🔍 Why did the model decide this?")
    st.caption("Red bars push towards FRAUD. Blue bars push towards LEGITIMATE.")
    try:
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(input_df)
        fig, ax   = plt.subplots(figsize=(8, 4))
        shap.waterfall_plot(
            shap.Explanation(
                values        = shap_vals[0],
                base_values   = explainer.expected_value,
                data          = input_df.values[0],
                feature_names = feature_names
            ),
            show=False, max_display=10
        )
        st.pyplot(fig)
        plt.close()
    except Exception as e:
        st.info("Adjust the sliders to see the SHAP explanation update.")

with col_right:
    st.subheader("📈 Model Performance")
    st.dataframe(pd.DataFrame({
        "Metric": ["PR-AUC", "ROC-AUC", "Fraud Recall", "Precision"],
        "Score":  ["0.877",  "0.979",   "89%",          "73%"]
    }), hide_index=True, use_container_width=True)

    st.subheader("🏆 Model Comparison")
    st.dataframe(pd.DataFrame({
        "Model":  ["XGBoost ✅", "Random Forest", "LightGBM", "Logistic Reg"],
        "PR-AUC": ["0.877",     "0.868",         "0.808",    "0.725"]
    }), hide_index=True, use_container_width=True)

    st.subheader("⚖️ Threshold Guide")
    st.dataframe(pd.DataFrame({
        "Threshold": ["0.1 - 0.3",     "0.5 (default)", "0.7 - 0.9"],
        "Effect":    ["Catch more fraud", "Balanced",    "Fewer false alarms"]
    }), hide_index=True, use_container_width=True)

st.divider()
st.caption("Built by Senuri Ganegoda  |  XGBoost + SMOTE + SHAP  |  Kaggle ULB Credit Card Fraud Dataset")
