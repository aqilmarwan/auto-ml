import argparse
import os
from typing import Tuple

import joblib
import numpy as np
import pandas as pd

from preprocess import transform_new_data, _feature_engineering, _coerce_dates, get_project_root
from utils import load_models


DEFAULT_MODEL = "XGBoost_calibrated"


def _compute_rule_flags(df_feat: pd.DataFrame) -> pd.DataFrame:
    """Generate simple, interpretable flags for investigators."""
    flags = pd.DataFrame(index=df_feat.index)
    flags["late_report"] = df_feat.get("report_delay_days", pd.Series(index=df_feat.index, dtype=float)).fillna(0) > 7
    flags["high_claim_to_premium"] = df_feat.get("claim_to_premium_ratio", 0).fillna(0) > 1.5
    flags["high_repair_to_value"] = df_feat.get("repair_to_value_ratio", 0).fillna(0) > 1.0
    flags["missing_police_report"] = df_feat.get("police_report_missing_flag", False).astype(bool)
    flags["missing_property_damage"] = df_feat.get("property_damage_missing_flag", False).astype(bool)
    flags["weekend_incident"] = df_feat.get("is_weekend", False).astype(bool)
    return flags


def score_file(input_path: str, model_name: str = DEFAULT_MODEL, output_path: str = None) -> Tuple[pd.DataFrame, str]:
    """Score new claims file and return predictions with risk flags."""
    project_root = get_project_root()
    if output_path is None:
        output_path = os.path.join(project_root, "models", "inference_output.csv")

    preprocessors = joblib.load(os.path.join(project_root, "models", "preprocessors.pkl"))
    models = load_models()

    if model_name not in models:
        raise ValueError(f"Model '{model_name}' not found. Available: {list(models.keys())}")

    model = models[model_name]

    raw_df = pd.read_csv(input_path)
    engineered_df = _feature_engineering(_coerce_dates(raw_df.copy()))
    flags = _compute_rule_flags(engineered_df)

    X = transform_new_data(engineered_df, preprocessors)
    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(X)[:, 1]
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
    else:
        scores = model.predict(X)

    preds = (scores >= 0.5).astype(int)

    output_df = raw_df.copy()
    output_df["fraud_risk_score"] = scores
    output_df["fraud_prediction"] = preds
    output_df = pd.concat([output_df, flags], axis=1)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    output_df.to_csv(output_path, index=False)
    print(f"Inference complete. Saved predictions to {output_path}")
    return output_df, output_path


def main():
    parser = argparse.ArgumentParser(description="Score new insurance claims for fraud risk.")
    parser.add_argument("--input", required=True, help="Path to CSV file with claim records.")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Model name to use (default: XGBoost_calibrated).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output CSV path (default: models/inference_output.csv)",
    )
    args = parser.parse_args()
    score_file(args.input, model_name=args.model, output_path=args.output)


if __name__ == "__main__":
    main()

