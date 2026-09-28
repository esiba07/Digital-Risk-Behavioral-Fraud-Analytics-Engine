# 🛡️ Digital Risk & Behavioral Fraud Analytics Engine
*An end-to-end fraud detection and velocity monitoring pipeline built on the IEEE-CIS benchmark dataset.*

## 📌 Project Overview
Online fraud in digital transactions and high-velocity environments moves fast, characterized by extreme class imbalance (fraud rates typically < 3%) and complex device-to-account relationships. This project implements a production-grade data pipeline designed to flag malicious behavior, calculate transaction velocity, and prevent financial loss while minimizing friction for legitimate users.

---

## 🛠️ Tech Stack & Architecture
* **Data Processing & Feature Engineering:** Python, Pandas, NumPy
* **Machine Learning:** LightGBM, Scikit-Learn
* **Validation Strategy:** Chronological Time-Series Splitting (Preventing data leakage)
* **Evaluation Metrics:** AUC-ROC, Precision-Recall tradeoff matrices

---

## 🔍 Key Features Engineered
1. **Behavioral Velocity Metrics:** Tracked rolling transaction frequencies and amounts grouped by anonymized user card fingerprints (`card1`) to detect sudden spending spikes.
2. **Amount Deviations:** Calculated log transformations and ratios comparing current transaction amounts against historical user averages.
3. **Identity & Network Risk Flags:** Isolated missing device metadata and suspicious email domain indicators commonly linked to multi-accounting and botnets.

---

## 📊 Model Performance & Fraud Ops Impact
* **Validation AUC-ROC:** Achieved high predictive separation on chronological validation folds.
* **Minimizing False Positives:** Tuned decision thresholds (`scale_pos_weight` and probability cutoffs) specifically to protect genuine customer checkout experiences from unnecessary blocks.
* **Operational Actionability:** Structured model outputs to prioritize high-risk queues for Fraud Operations analysts.
