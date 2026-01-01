import os
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

try:
    from xgboost import XGBClassifier
except ImportError as e:
    if "libomp.dylib" in str(e) or "OpenMP" in str(e):
        print("\n" + "=" * 70)
        print("XGBoost requires OpenMP runtime library on macOS.")
        print("Please install it by running: brew install libomp")
        print("=" * 70 + "\n")
    raise

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

from preprocess import preprocess_data
from utils import save_models

RANDOM_STATE = 42


def _cv_score_auc(model, X, y):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
    return scores.mean()


def train_models():
    """
    Train and compare multiple ML models for fraud detection.
    Returns trained models and CV AUC scores.
    """
    print("Loading and preprocessing data...")
    data = preprocess_data()
    X_train = data["X_train_smote"]
    y_train = data["y_train_smote"]
    X_val = data["X_val"]
    y_val = data["y_val"]
    class_weight = data["preprocessors"]["class_weight"]

    print(f"Training set shape (SMOTE): {X_train.shape}")
    print(f"Validation set shape: {X_val.shape}")

    models = {}
    cv_scores = {}
    val_scores = {}

    # Model definitions
    model_specs = {
        "Logistic Regression": LogisticRegression(
            random_state=RANDOM_STATE, max_iter=2000, class_weight=class_weight, solver="liblinear"
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=1,
            class_weight=class_weight,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "Decision Tree": DecisionTreeClassifier(
            class_weight=class_weight,
            random_state=RANDOM_STATE,
        ),
        "Support Vector Machine": SVC(
            probability=True,
            class_weight=class_weight,
            random_state=RANDOM_STATE,
        ),
        "AdaBoost": AdaBoostClassifier(
            n_estimators=200,
            learning_rate=0.1,
            random_state=RANDOM_STATE,
        ),
        "Naive Bayes": GaussianNB(),
    }

    # XGBoost with imbalance handling
    scale_pos_weight = (y_train == 0).sum() / max(1, (y_train == 1).sum())
    model_specs["XGBoost"] = XGBClassifier(
        objective="binary:logistic",
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        eval_metric="logloss",
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=-1,
    )

    # Optional LightGBM
    if HAS_LGBM:
        model_specs["LightGBM"] = LGBMClassifier(
            objective="binary",
            class_weight=class_weight,
            random_state=RANDOM_STATE,
            n_estimators=300,
            learning_rate=0.05,
            max_depth=-1,
            subsample=0.8,
            colsample_bytree=0.8,
        )

    # Train and evaluate
    for name, model in model_specs.items():
        print(f"\nTraining {name}...")
        try:
            cv_auc = _cv_score_auc(model, X_train, y_train)
            cv_scores[name] = cv_auc
        except Exception as ex:
            warnings.warn(f"CV failed for {name}: {ex}")
            cv_scores[name] = np.nan

        model.fit(X_train, y_train)
        models[name] = model

        # Validation ROC-AUC on non-SMOTE set
        try:
            val_proba = model.predict_proba(X_val)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X_val)
            val_auc = roc_auc_score(y_val, val_proba)
            val_scores[name] = val_auc
        except Exception:
            val_scores[name] = np.nan

    # Choose best model by validation ROC-AUC
    best_model_name = max(val_scores, key=lambda k: val_scores[k] if not np.isnan(val_scores[k]) else -np.inf)
    print("\n" + "=" * 60)
    print(f"Best model by validation ROC-AUC: {best_model_name} ({val_scores[best_model_name]:.4f})")
    print("=" * 60)

    # Calibrate best model probabilities on validation set (sigmoid)
    best_model = model_specs[best_model_name]
    # sklearn >=1.4 uses 'estimator' instead of deprecated 'base_estimator'
    calibrator = CalibratedClassifierCV(estimator=clone(best_model), method="sigmoid", cv=5)
    calibrator.fit(pd.concat([X_train, X_val]), np.concatenate([y_train, y_val]))
    models[f"{best_model_name}_calibrated"] = calibrator

    # Print CV results
    print("\n" + "=" * 50)
    print("Cross-Validation ROC-AUC Results:")
    print("=" * 50)
    for model_name, score in sorted(cv_scores.items(), key=lambda x: (x[1] if not np.isnan(x[1]) else -np.inf), reverse=True):
        print(f"{model_name:30s}: {score:.4f}")

    # Save models
    print("\nSaving models...")
    save_models(models)

    return models, cv_scores


if __name__ == "__main__":
    train_models()

