import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from utils import plot_confusion_matrix, load_models, get_project_root
from preprocess import preprocess_data


TOP_K_PCT = 0.1


def _precision_recall_lift_at_k(y_true: np.ndarray, y_scores: np.ndarray, k_pct: float = TOP_K_PCT):
    n = len(y_true)
    k = max(1, int(n * k_pct))
    order = np.argsort(y_scores)[::-1][:k]
    y_top = y_true[order]
    precision_k = y_top.mean()
    recall_k = y_top.sum() / y_true.sum() if y_true.sum() > 0 else 0
    baseline = y_true.mean() if y_true.mean() > 0 else 1e-8
    lift_k = precision_k / baseline
    return precision_k, recall_k, lift_k, k


def _shap_summary(model_name: str, model, X: pd.DataFrame):
    project_root = get_project_root()
    models_dir = os.path.join(project_root, "models")
    os.makedirs(models_dir, exist_ok=True)
    try:
        explainer = shap.TreeExplainer(model)
        sample = X.sample(min(500, len(X)), random_state=42)
        shap_values = explainer.shap_values(sample)
        shap.summary_plot(
            shap_values,
            sample,
            plot_type="bar",
            show=False,
            max_display=20,
        )
        plt.tight_layout()
        plt.savefig(os.path.join(models_dir, f"shap_summary_{model_name.replace(' ', '_')}.png"), dpi=300, bbox_inches="tight")
        plt.close()
    except Exception as ex:
        warnings.warn(f"SHAP failed for {model_name}: {ex}")


def evaluate_models():
    """
    Evaluate all trained models with rich metrics, calibration, and explainability.
    """
    print("Loading models and test data...")
    models = load_models()
    data = preprocess_data(save_preprocessor=False)
    X_test = data["X_test"]
    y_test = data["y_test"]

    results = []

    print("\n" + "=" * 80)
    print("MODEL EVALUATION RESULTS")
    print("=" * 80)

    for model_name, model in models.items():
        print(f"\nEvaluating {model_name}...")

        # Probabilities
        if hasattr(model, "predict_proba"):
            y_scores = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            y_scores = model.decision_function(X_test)
        else:
            y_scores = model.predict(X_test)

        y_pred = (y_scores >= 0.5).astype(int)

        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_scores)
        pr_auc = average_precision_score(y_test, y_scores)
        brier = brier_score_loss(y_test, y_scores)
        precision_k, recall_k, lift_k, k_val = _precision_recall_lift_at_k(y_test, y_scores, TOP_K_PCT)

        results.append(
            {
                "Model": model_name,
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall,
                "F1": f1,
                "ROC_AUC": roc_auc,
                "PR_AUC": pr_auc,
                "Brier": brier,
                "Precision@k": precision_k,
                "Recall@k": recall_k,
                "Lift@k": lift_k,
                "k": k_val,
            }
        )

        # Log metrics
        print(f"  ROC AUC:     {roc_auc:.4f}")
        print(f"  PR  AUC:     {pr_auc:.4f}")
        print(f"  Accuracy:    {accuracy:.4f}")
        print(f"  Precision:   {precision:.4f}")
        print(f"  Recall:      {recall:.4f}")
        print(f"  F1:          {f1:.4f}")
        print(f"  Brier score: {brier:.4f}")
        print(f"  Precision@{TOP_K_PCT:.0%}: {precision_k:.4f} | Recall@{TOP_K_PCT:.0%}: {recall_k:.4f} | Lift@k: {lift_k:.2f}")

        # Confusion matrix
        project_root = get_project_root()
        cm_path = os.path.join(project_root, "models", f"confusion_matrix_{model_name.replace(' ', '_')}.png")
        plot_confusion_matrix(y_test, y_pred, model_name, save_path=cm_path)

        # SHAP for tree models
        if any(tag in model_name.lower() for tag in ["xgboost", "random forest", "lightgbm"]) and X_test.shape[1] > 1:
            _shap_summary(model_name, model, X_test)

    # Results DataFrame
    results_df = pd.DataFrame(results).sort_values("ROC_AUC", ascending=False)

    print("\n" + "=" * 80)
    print("PERFORMANCE SUMMARY (Sorted by ROC_AUC)")
    print("=" * 80)
    print(results_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Comparison plot (ROC AUC)
    plt.figure(figsize=(12, 6))
    model_names = results_df["Model"].values
    aucs = results_df["ROC_AUC"].values
    bars = plt.bar(range(len(model_names)), aucs, color="steelblue")
    plt.xlabel("Model", fontsize=12)
    plt.ylabel("ROC AUC", fontsize=12)
    plt.title("Model ROC AUC Comparison", fontsize=14, fontweight="bold")
    plt.xticks(range(len(model_names)), model_names, rotation=45, ha="right")
    plt.ylim([0, 1])
    plt.grid(axis="y", alpha=0.3)

    for bar, auc in zip(bars, aucs):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"{auc:.3f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    project_root = get_project_root()
    accuracy_path = os.path.join(project_root, "models", "model_auc_comparison.png")
    plt.savefig(accuracy_path, dpi=300, bbox_inches="tight")
    print(f"\nAUC comparison chart saved to {accuracy_path}")
    plt.close()

    # Save results to CSV
    results_path = os.path.join(project_root, "models", "evaluation_results.csv")
    results_df.to_csv(results_path, index=False)
    print(f"Evaluation results saved to {results_path}")

    return results_df


if __name__ == "__main__":
    evaluate_models()

