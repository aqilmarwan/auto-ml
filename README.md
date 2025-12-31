<div align="center">
  <h1>Motor Insurance Fraud Detection</h1>
  <p>End-to-end ML system to score motor insurance claims for fraud risk, combining research-backed features, imbalance handling (SMOTE + class weights), calibrated tree models, and explainability.</p>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.x-blue?style=flat-square&logo=python&logoColor=white" alt="Python" /></a>
  <a href="https://scikit-learn.org/"><img src="https://img.shields.io/badge/scikit--learn-ML-orange?style=flat-square&logo=scikit-learn&logoColor=white" alt="sklearn" /></a>
  <a href="https://www.kaggle.com/datasets/buntyshah/auto-insurance-claims-data"><img src="https://img.shields.io/badge/Dataset-Kaggle-lightgrey?style=flat-square&logo=kaggle" alt="dataset" /></a>
</div>

> Note: Built for a fraud analytics assessment using the Kaggle Auto Insurance Claims dataset and aligned with the IJCA paper (Njeru et al., 2025). Data is synthetic/simplified—do not deploy without validating on real data.

---

## Table of Contents
- Overview
- Data
- Research alignment
- Features
- Models
- Evaluation
- Quickstart
- Inference
- Streamlit app
- Limitations

---

## Overview
Fraudulent motor claims (Own Damage and Third-Party Bodily Injury) are rare and costly. This project trains multiple classifiers, handles imbalance with SMOTE and class weights, calibrates probabilities, and surfaces explainability (SHAP) plus simple rule flags to support investigators.

## Data
- Source: Kaggle Auto Insurance Claims ([link](https://www.kaggle.com/datasets/buntyshah/auto-insurance-claims-data))
- Target: `fraud_reported` (binary, yes/no)
- Scope: policyholder, policy, incident, vehicle, injury/property/vehicle claim amounts, police report availability
- Synthetic dataset; missing values treated as signal, not noise

## Research alignment
- Mirrors IJCA paper (Njeru et al., 2025) with supervised learning under imbalance, SMOTE, and tree-based models.
- Adds calibrated probabilities, richer feature engineering, top-k business metrics, and SHAP explainability.

## Features (engineered in `src/preprocess.py`)
- Temporal/behavioral: `report_delay_days`, `policy_tenure_years`, `claims_per_year`
- Financial ratios: `claim_to_premium_ratio`, injury/property/vehicle claim shares, `repair_to_value_ratio`
- Missingness flags: `police_report_missing_flag`, `property_damage_missing_flag`
- Geo/time: `incident_state`, `time_of_day_bucket`, `is_weekend`
- Cleaning: safe coercion of dates/numerics, high-cardinality ID drops (`policy_number`, `insured_zip`, `incident_location`)

## Models (trained in `src/train.py`)
- Logistic Regression (class-weighted baseline)
- Random Forest, Decision Tree, SVM, AdaBoost, Naive Bayes
- XGBoost (scale_pos_weight) and optional LightGBM (class_weight)
- Best model is chosen by validation ROC AUC, then probability-calibrated (sigmoid) and saved as `<model>_calibrated`

## Evaluation (in `src/evaluate.py`)
- Metrics: ROC AUC, PR AUC, accuracy, precision, recall, F1, Brier
- Business focus: precision@k, recall@k, lift@k (k = top 10% by risk)
- Artifacts: confusion matrices, ROC AUC comparison bar plot, SHAP bar summaries for tree models, `models/evaluation_results.csv`

## Quickstart
```bash
pip install -r requirements.txt
python src/train.py          # preprocess, SMOTE on train only, train + calibrate, save models
python src/evaluate.py       # evaluate all saved models, save metrics/plots
```

## Inference (batch scoring)
```bash
python src/inference.py --input data/insurance_claims.csv --model XGBoost_calibrated
```
- Outputs `models/inference_output.csv` with `fraud_risk_score`, `fraud_prediction`, and rule flags (`late_report`, `high_claim_to_premium`, `missing_police_report`, etc.).
- Reuses saved preprocessors (`models/preprocessors.pkl`) to ensure train/inference parity.

## Streamlit app
```bash
streamlit run src/app.py
```
- Upload CSV, select model (AdaBoost/XGBoost variants), view fraud vs genuine counts, table of predictions, and download results.

## Limitations
- Synthetic dataset; real-world telematics, networks, and notes are absent.
- Rule flags are simple heuristics; thresholds may need tuning for production.
- No automated data/label drift monitoring; add before deployment.