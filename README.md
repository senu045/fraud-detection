# 💳 Credit Card Fraud Detection Engine

[![Live Demo](https://img.shields.io/badge/🚀_Live_Demo-Streamlit-FF4B4B?style=for-the-badge)](https://fraud-detection-ehv4snw5r7xevgqelz5zmn.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-Best_Model-orange?style=for-the-badge)](https://xgboost.readthedocs.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

> **An end-to-end machine learning system that detects fraudulent credit card transactions — with full explainability, threshold tuning, and a live interactive demo.**

---

## 🌐 Live Demo

👉 **[Click here to open the live app](https://fraud-detection-ehv4snw5r7xevgqelz5zmn.streamlit.app/)**

- Input a transaction amount and time
- Adjust the decision threshold
- See real-time fraud probability + SHAP explanation

---

## 📌 Project Overview

Credit card fraud is a critical real-world problem — only **0.17%** of transactions are fraudulent, making this an extreme class imbalance challenge. A naive model that predicts every transaction as legitimate achieves 99.83% accuracy but catches **zero fraud**.

This project solves that using:
- **SMOTE** to synthetically balance the training data
- **XGBoost** as the best-performing model (PR-AUC: 0.877)
- **SHAP** to explain every single prediction
- **Threshold tuning** to control the precision-recall trade-off
- **Streamlit** to deliver a live, interactive web application

---

## 🏆 Model Results

| Model | PR-AUC | ROC-AUC |
|-------|--------|---------|
| **XGBoost ✅ (Best)** | **0.877** | **0.979** |
| Random Forest | 0.868 | 0.969 |
| LightGBM | 0.808 | 0.969 |
| Logistic Regression | 0.725 | 0.970 |

> PR-AUC is used instead of accuracy because accuracy is misleading on imbalanced datasets.

### Confusion Matrix (XGBoost @ threshold 0.5)
| | Predicted Legit | Predicted Fraud |
|--|--|--|
| **Actual Legit** | 56,832 ✅ | 32 ❌ |
| **Actual Fraud** | 11 ❌ | 87 ✅ |

- **89% of all fraud cases caught**
- Only **32 false alarms** out of 56,864 legitimate transactions

---

## 🔍 Key Technical Highlights

### 1. Class Imbalance — Why Accuracy Fails
Only 492 fraud cases out of 284,807 transactions (0.17%). Solved with **SMOTE** — synthetic minority oversampling applied only to training data to prevent data leakage.

### 2. SHAP Explainability
Every prediction comes with a SHAP waterfall chart explaining exactly which features pushed the model towards or away from fraud — making the model transparent and trustworthy for business stakeholders.

### 3. Threshold Tuning
The decision threshold is fully adjustable in the app:
- **Lower threshold (0.1–0.3):** Catch more fraud, more false alarms
- **Default (0.5):** Balanced precision and recall
- **Higher threshold (0.7–0.9):** Fewer false alarms, miss some fraud

### 4. Feature Importance
Top fraud-predicting features (from SHAP & correlation analysis):
`V14` > `V17` > `V12` > `V10` > `V16`

---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| Language | Python 3.12 |
| ML Models | XGBoost, LightGBM, Random Forest, Logistic Regression |
| Imbalance | imbalanced-learn (SMOTE) |
| Explainability | SHAP |
| Visualization | Matplotlib, Seaborn |
| App Framework | Streamlit |
| Data | Pandas, NumPy |
| Dataset | Kaggle ULB Credit Card Fraud |

---

## 📂 Project Structure
