<div align="center">
  <h1>Motor Insurance Fraud Detection</h1>
  <p>End-to-end ML system to score motor insurance claims for fraud risk.</p>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.x-blue?style=flat-square&logo=python&logoColor=white" alt="Python" /></a>
  <a href="https://scikit-learn.org/"><img src="https://img.shields.io/badge/scikit--learn-ML-orange?style=flat-square&logo=scikit-learn&logoColor=white" alt="sklearn" /></a>
  <a href="https://www.kaggle.com/datasets/buntyshah/auto-insurance-claims-data"><img src="https://img.shields.io/badge/Dataset-Kaggle-lightgrey?style=flat-square&logo=kaggle" alt="dataset" /></a>
</div>

> [!NOTE]
> Built for a fraud analytics assessment using the Kaggle's dataset and aligned with the IJCA paper [Njeru et al., 2025](https://www.ijcaonline.org/archives/volume187/number65/njeru-2025-ijca-926105.pdf).

---

## Overview
Fraudulent motor claims (Own Damage and Third-Party Bodily Injury) are rare and costly. This project trains multiple classifiers, handles imbalance with SMOTE and class weights, calibrates probabilities, and surfaces explainability (SHAP) plus simple rule flags to support investigators.

## Data
- Source: Kaggle Auto Insurance Claims ([link](https://www.kaggle.com/datasets/buntyshah/auto-insurance-claims-data))
- Target: `fraud_reported` (yes/no)
- Scope: policyholder, policy, incident, vehicle, injury/property/vehicle claim amounts, police report availability
- Synthetic dataset; missing values treated as signal, not noise

## Research alignment
- Reimplemented from IJCA paper (Njeru et al., 2025) with supervised learning under imbalance, SMOTE, and tree-based models.
- Added calibrated probabilities, richer feature engineering, top-k business metrics, and SHAP explainability can be found in `/models`.

## Features (engineered in `src/preprocess.py`)
- Temporal/behavioral: `report_delay_days`, `policy_tenure_years`, `claims_per_year`
- Financial ratios: `claim_to_premium_ratio`, injury/property/vehicle claim shares, `repair_to_value_ratio`
- Missingness flags: `police_report_missing_flag`, `property_damage_missing_flag`
- Geo/time: `incident_state`, `time_of_day_bucket`, `is_weekend`
- Cleaning: safe coercion of dates/numerics, high-cardinality ID drops (`policy_number`, `insured_zip`, `incident_location`)

### Feature Engineering Suggestions
- Geo enrichment: map `incident_city` and `incident_location` to latitude proxies (`city_lat`, `location_lat`) and combine them into a single geo signal to capture regional risk patterns.
![Geo enrichment](public/d1.png)
- Claim efficiency: compare `total_claim_amount` to `policy_annual_premium` via `claim_to_premium_ratio` to flag claims that are unusually large relative to the premium paid.
![Claim-to-premium ratio](public/d2.png)
- Reporting timeliness: compute `report_delay_days` from `incident_date` and `incident_reported` to capture lag risk—long delays can correlate with higher fraud likelihood.
![Report delay](public/d3.png)

## Assessment Questions
1.
2.
3.
4.
5.
6.

## Quickstart
```bash
pip install -r requirements.txt
python src/train.py         
python src/evaluate.py      
```

## Inference (batch scoring)
```bash
python src/inference.py --input data/insurance_claims.csv --model XGBoost_calibrated
```
- Outputs `models/inference_output.csv` with `fraud_risk_score`, `fraud_prediction`, and rule flags (`late_report`, `high_claim_to_premium`, `missing_police_report`, etc.).
- Reuses saved preprocessors (`models/preprocessors.pkl`) to ensure train/inference parity.

## Limitations
- Synthetic dataset; real-world telematics, networks, and notes are absent.
- Rule flags are simple heuristics; thresholds may need tuning for production.
- No automated data/label drift monitoring; add before deployment.