import os
from datetime import datetime
from typing import Dict, Tuple, List, Any

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight


RANDOM_STATE = 42


def get_project_root():
    """Get the project root directory."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(current_dir)


def _coerce_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Parse date columns if present."""
    for col in ["incident_date", "policy_bind_date", "claim_date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def _feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Create engineered features aligned with the paper + enhancements."""
    df = df.copy()

    # Normalize column names used downstream
    if "policy_deductable" in df.columns and "policy_deductible" not in df.columns:
        df["policy_deductible"] = df["policy_deductable"]

    # Basic totals
    for col in ["injury_claim", "property_claim", "vehicle_claim", "policy_annual_premium", "policy_deductible"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["total_claim_amount"] = (
        df.get("injury_claim", 0)
        + df.get("property_claim", 0)
        + df.get("vehicle_claim", 0)
    )

    # Temporal / behavioral
    if "incident_date" in df.columns and "claim_date" in df.columns:
        df["report_delay_days"] = (df["claim_date"] - df["incident_date"]).dt.days
    elif "incident_date" in df.columns and "policy_bind_date" in df.columns:
        # fallback proxy: time between policy start and incident
        df["report_delay_days"] = (df["incident_date"] - df["policy_bind_date"]).dt.days
    else:
        df["report_delay_days"] = np.nan

    df["policy_tenure_years"] = np.where(
        "months_as_customer" in df.columns,
        df["months_as_customer"] / 12.0,
        np.nan,
    )
    tenure_safe = df["policy_tenure_years"].replace(0, np.nan)
    df["claims_per_year"] = df["total_claim_amount"] / tenure_safe

    # Financial ratios
    df["claim_to_premium_ratio"] = df["total_claim_amount"] / (df["policy_annual_premium"].replace(0, np.nan))
    df["injury_claim_share"] = df.get("injury_claim", 0) / (df["total_claim_amount"].replace(0, np.nan))
    df["property_claim_share"] = df.get("property_claim", 0) / (df["total_claim_amount"].replace(0, np.nan))
    df["vehicle_claim_share"] = df.get("vehicle_claim", 0) / (df["total_claim_amount"].replace(0, np.nan))
    df["repair_to_value_ratio"] = df.get("vehicle_claim", 0) / (
        (df.get("policy_annual_premium", 0) + df.get("policy_deductible", 0)).replace(0, np.nan)
    )

    # Missingness as signal
    for col, flag in [("police_report_available", "police_report_missing_flag"), ("property_damage", "property_damage_missing_flag")]:
        if col in df.columns:
            df[flag] = df[col].isna() | (df[col].astype(str).str.strip() == "?")
        else:
            df[flag] = True

    # Geo/time buckets
    if "incident_hour_of_the_day" in df.columns:
        hour = pd.to_numeric(df["incident_hour_of_the_day"], errors="coerce")
        df["time_of_day_bucket"] = pd.cut(
            hour,
            bins=[-1, 6, 12, 18, 24],
            labels=["night", "morning", "afternoon", "evening"]
        )
    else:
        df["time_of_day_bucket"] = "unknown"

    if "incident_date" in df.columns:
        df["is_weekend"] = df["incident_date"].dt.dayofweek >= 5
    else:
        df["is_weekend"] = False

    # Clean up infs from ratios
    ratio_cols = [
        "claim_to_premium_ratio",
        "injury_claim_share",
        "property_claim_share",
        "vehicle_claim_share",
        "repair_to_value_ratio",
        "claims_per_year",
    ]
    for col in ratio_cols:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)

    return df


def _encode_categoricals(
    X_train: pd.DataFrame,
    X_other: List[pd.DataFrame],
    categorical_cols: List[str]
) -> Tuple[pd.DataFrame, List[pd.DataFrame], Dict[str, LabelEncoder]]:
    """Label encode categorical columns fitted on train only."""
    encoders: Dict[str, LabelEncoder] = {}

    def _transform_with_encoder(series: pd.Series, encoder: LabelEncoder) -> pd.Series:
        values = series.astype(str).fillna("nan")
        unknown_mask = ~values.isin(encoder.classes_)
        if unknown_mask.any():
            encoder.classes_ = np.append(encoder.classes_, "unk")
            values = values.where(~unknown_mask, "unk")
        return pd.Series(encoder.transform(values), index=series.index)

    X_train_enc = X_train.copy()
    others_enc = [df.copy() for df in X_other]

    for col in categorical_cols:
        le = LabelEncoder()
        X_train_enc[col] = le.fit_transform(X_train_enc[col].astype(str).fillna("nan"))
        encoders[col] = le
        for df_enc in others_enc:
            if col in df_enc.columns:
                df_enc[col] = _transform_with_encoder(df_enc[col], le)

    return X_train_enc, others_enc, encoders


def preprocess_data(data_path=None, save_preprocessor=True):
    """
    Preprocess insurance claims data for fraud detection.
    - Loads Kaggle auto insurance claims data
    - Engineers temporal/financial/behavioral features
    - Splits into train/val/test (time-aware if dates exist)
    - Encodes categoricals, scales numerics (fit on train only)
    - Applies SMOTE on train only
    
    Returns:
        dict with train/val/test splits, resampled train, and metadata
    """
    if data_path is None:
        project_root = get_project_root()
        data_path = os.path.join(project_root, "data", "insurance_claims.csv")

    df_raw = pd.read_csv(data_path)
    df_raw = df_raw.replace("?", np.nan)
    df_raw = _coerce_dates(df_raw)
    df = _feature_engineering(df_raw)

    # Target
    if "fraud_reported" not in df.columns:
        raise ValueError("Expected 'fraud_reported' column in dataset.")
    y = df["fraud_reported"].astype(str)
    target_encoder = LabelEncoder()
    y_encoded = target_encoder.fit_transform(y)

    # Drop high-cardinality IDs and leakage-prone columns
    drop_cols = [col for col in ["policy_number", "insured_zip", "incident_location"] if col in df.columns]
    feature_df = df.drop(columns=["fraud_reported"] + drop_cols, errors="ignore")

    # Train/val/test split (time-aware if incident_date present)
    if "incident_date" in feature_df.columns and feature_df["incident_date"].notna().any():
        sorted_idx = feature_df["incident_date"].argsort()
        feature_df = feature_df.iloc[sorted_idx].reset_index(drop=True)
        y_encoded = y_encoded[sorted_idx]
        test_size = max(1, int(len(feature_df) * 0.2))
        df_trainval = feature_df.iloc[:-test_size]
        y_trainval = y_encoded[:-test_size]
        X_test = feature_df.iloc[-test_size:]
        y_test = y_encoded[-test_size:]
        X_train, X_val, y_train, y_val = train_test_split(
            df_trainval, y_trainval, test_size=0.2, random_state=RANDOM_STATE, stratify=y_trainval
        )
    else:
        X_temp, X_test, y_temp, y_test = train_test_split(
            feature_df, y_encoded, test_size=0.2, random_state=RANDOM_STATE, stratify=y_encoded
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.2, random_state=RANDOM_STATE, stratify=y_temp
        )

    # Remove raw date columns post-split; copy to avoid chained assignment warnings
    drop_date_cols = ["incident_date", "policy_bind_date", "claim_date"]
    X_train = X_train.drop(columns=drop_date_cols, errors="ignore").copy()
    X_val = X_val.drop(columns=drop_date_cols, errors="ignore").copy()
    X_test = X_test.drop(columns=drop_date_cols, errors="ignore").copy()

    # Identify columns
    categorical_cols = X_train.select_dtypes(include=["object", "bool", "category"]).columns.tolist()
    numerical_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()

    # Impute numeric missing values using train medians (fallback to 0 if all NaN)
    numeric_impute_values = X_train[numerical_cols].median().fillna(0)
    for df_split in [X_train, X_val, X_test]:
        df_split[numerical_cols] = df_split[numerical_cols].fillna(numeric_impute_values)

    # Encode categoricals (fit on train)
    X_train_enc, [X_val_enc, X_test_enc], label_encoders = _encode_categoricals(
        X_train, [X_val, X_test], categorical_cols
    )

    # Scale numerics (fit on train)
    scaler = StandardScaler()
    X_train_enc[numerical_cols] = scaler.fit_transform(X_train_enc[numerical_cols])
    X_val_enc[numerical_cols] = scaler.transform(X_val_enc[numerical_cols])
    X_test_enc[numerical_cols] = scaler.transform(X_test_enc[numerical_cols])

    # SMOTE on train only
    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_enc, y_train)

    # Class weights for imbalance-aware models
    classes = np.unique(y_train)
    class_weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weight_dict = {cls: w for cls, w in zip(classes, class_weights)}

    preprocessors = {
        "label_encoders": label_encoders,
        "scaler": scaler,
        "target_encoder": target_encoder,
        "categorical_cols": categorical_cols,
        "numerical_cols": numerical_cols,
        "feature_columns": X_train_enc.columns.tolist(),
        "numeric_impute_values": numeric_impute_values,
        "class_weight": class_weight_dict,
    }

    if save_preprocessor:
        project_root = get_project_root()
        models_dir = os.path.join(project_root, "models")
        os.makedirs(models_dir, exist_ok=True)
        preprocessors_path = os.path.join(models_dir, "preprocessors.pkl")
        joblib.dump(preprocessors, preprocessors_path)

    return {
        "X_train": X_train_enc,
        "y_train": y_train,
        "X_val": X_val_enc,
        "y_val": y_val,
        "X_test": X_test_enc,
        "y_test": y_test,
        "X_train_smote": X_train_smote,
        "y_train_smote": y_train_smote,
        "preprocessors": preprocessors,
    }


def transform_new_data(df: pd.DataFrame, preprocessors: Dict[str, Any]) -> pd.DataFrame:
    """
    Apply saved preprocessing (encoding + scaling) to new data for inference.
    """
    df_proc = _feature_engineering(_coerce_dates(df)).copy()
    df_proc = df_proc.drop(columns=["fraud_reported", "policy_number", "insured_zip", "incident_location"], errors="ignore")

    categorical_cols = preprocessors["categorical_cols"]
    numerical_cols = preprocessors["numerical_cols"]
    feature_columns = preprocessors["feature_columns"]
    encoders = preprocessors["label_encoders"]
    scaler = preprocessors["scaler"]

    # Ensure missing engineered columns exist
    for col in feature_columns:
        if col not in df_proc.columns:
            df_proc[col] = np.nan

    df_proc = df_proc[feature_columns].copy()

    # Impute numerics to mirror training pipeline
    numeric_impute_values = preprocessors.get("numeric_impute_values")
    if numeric_impute_values is not None:
        df_proc[numerical_cols] = df_proc[numerical_cols].fillna(numeric_impute_values)
    else:
        df_proc[numerical_cols] = df_proc[numerical_cols].fillna(0)

    # Encode categoricals using saved encoders
    for col in categorical_cols:
        if col in df_proc.columns:
            le: LabelEncoder = encoders[col]
            values = df_proc[col].astype(str).fillna("nan")
            unknown_mask = ~values.isin(le.classes_)
            if unknown_mask.any():
                le.classes_ = np.append(le.classes_, "unk")
                values = values.where(~unknown_mask, "unk")
            df_proc[col] = le.transform(values)

    # Scale numerics
    df_proc[numerical_cols] = scaler.transform(df_proc[numerical_cols])
    return df_proc

