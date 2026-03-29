"""
=============================================================================
Phase 3.1: Model Evaluation, Validation, & Interpretability
=============================================================================
This module generates full analytical reporting on trained ML models including:
- Classification metrics (Precision, Recall, ROC-AUC)
- Regression metrics (MAE, RMSE, R²)
- Interpretability via SHAP (SHapley Additive exPlanations)
- Confusion Matrices & Precision-Recall curves

Outputs are persisted locally into `reports/evaluation/` and `reports/interpretability/`.
"""

import logging
import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("evaluate")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data" / "train_test"
REPORTS_DIR = PROJECT_ROOT / "reports"
EVAL_DIR = REPORTS_DIR / "evaluation"
INTERP_DIR = REPORTS_DIR / "interpretability"

for d in [EVAL_DIR, INTERP_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  1. CLASSIFICATION EVALUATION                                           ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def evaluate_classification_models(X_test: pd.DataFrame, y_test: pd.Series):
    """Evaluate Churn models and output classification metrics."""
    logger.info("--- Evaluating Classification Models ---")
    class_dir = MODELS_DIR / "classification"
    if not class_dir.exists():
        logger.warning(f"Classification paths missing: {class_dir}")
        return

    # Strictly transactional features as trained
    allowed_features = [
        "Frequency", "MonetaryTotal", "MonetaryAvg", "MonetaryStd", 
        "MonetaryMin", "MonetaryMax", "TotalQuantity", 
        "AvgQuantityPerTransaction", "MinQuantity", "MaxQuantity",
        "UniqueProducts", "UniqueDescriptions", "AvgProductsPerTransaction",
        "NegativeQuantityCount", "ZeroPriceCount", "CancelledTransactions",
        "ReturnRatio", "TotalTransactions", "UniqueInvoices", "AvgLinesPerInvoice",
        "Age", "SupportTicketsCount", "SatisfactionScore"
    ]
    keep_cols = [c for c in allowed_features if c in X_test.columns]
    X_eval = X_test[keep_cols]

    metrics_registry = {}
    best_model_name = None
    best_auc = 0.0

    plt.figure(figsize=(10, 8))
    
    for model_path in class_dir.glob("*.joblib"):
        model_name = model_path.stem
        model = joblib.load(model_path)
        
        # Predictions
        preds = model.predict(X_eval)
        probs = model.predict_proba(X_eval)[:, 1] if hasattr(model, "predict_proba") else preds
        
        # Calculate Metrics
        auc = roc_auc_score(y_test, probs)
        report = classification_report(y_test, preds, output_dict=True)
        cm = confusion_matrix(y_test, preds)

        metrics_registry[model_name] = {
            "Accuracy": report["accuracy"],
            "Precision_Class1": report["1"]["precision"],
            "Recall_Class1": report["1"]["recall"],
            "F1_Class1": report["1"]["f1-score"],
            "ROC_AUC": auc
        }
        
        logger.info(f"{model_name:>18} | AUC: {auc:.4f} | F1 (Churn): {report['1']['f1-score']:.4f}")

        # ROC Curve plotting
        fpr, tpr, _ = roc_curve(y_test, probs)
        plt.plot(fpr, tpr, label=f"{model_name} (AUC = {auc:.3f})")

        # Confusion Matrix Export
        plt_cm = plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
        plt.title(f"Confusion Matrix: {model_name}")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt_cm.savefig(EVAL_DIR / f"{model_name}_confusion_matrix.png", dpi=100)
        plt.close(plt_cm)

        # Track best model for SHAP context
        if auc > best_auc:
            best_auc = auc
            best_model_name = model_name

    # Finalize ROC Plot
    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.title("ROC/AUC Curve Comparison - Churn Classification")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.savefig(EVAL_DIR / "classification_roc_comparison.png", dpi=120)
    plt.close()

    # Save detailed JSON metrics
    with open(EVAL_DIR / "classification_metrics.json", "w") as f:
        json.dump(metrics_registry, f, indent=4)
        
    return best_model_name, X_eval


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  2. REGRESSION EVALUATION                                               ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def evaluate_regression_models(X_test: pd.DataFrame, y_test: pd.Series):
    """Evaluate Revenue models and output regression metrics."""
    logger.info("--- Evaluating Regression Models ---")
    reg_dir = MODELS_DIR / "regression"
    if not reg_dir.exists():
        return

    # Drop explicit target/leaks mathematically identical to Churn prevention module
    drop_cols = ["MonetaryTotal", "MonetaryPerDay", "AvgBasketValue", "MonetaryAvg", "MonetaryStd", "MonetaryMin", "MonetaryMax"]
    X_eval = X_test.drop(columns=[c for c in drop_cols if c in X_test.columns])

    metrics_registry = {}

    for model_path in reg_dir.glob("*.joblib"):
        model_name = model_path.stem
        model = joblib.load(model_path)
        
        preds = model.predict(X_eval)
        
        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)

        metrics_registry[model_name] = {
            "MAE": mae,
            "RMSE": rmse,
            "R2_Score": r2
        }
        
        logger.info(f"{model_name:>18} | R²: {r2:.4f} | MAE: {mae:.2f} | RMSE: {rmse:.2f}")

        # Scatter Plot Real vs Predictions
        plt_sc = plt.figure(figsize=(8, 6))
        plt.scatter(y_test, preds, alpha=0.3, color="green")
        max_val = max(y_test.max(), preds.max())
        plt.plot([0, max_val], [0, max_val], "r--")
        plt.title(f"Real vs Predicted Revenue - {model_name}")
        plt.xlabel("Actual Revenue")
        plt.ylabel("Predicted Revenue")
        plt_sc.savefig(EVAL_DIR / f"{model_name}_residuals.png", dpi=100)
        plt.close(plt_sc)

    # Save detailed JSON metrics
    with open(EVAL_DIR / "regression_metrics.json", "w") as f:
        json.dump(metrics_registry, f, indent=4)


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  3. AI GOVERNANCE & INTERPRETABILITY (SHAP)                             ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def generate_shap_explanations(best_model_name: str, X_eval: pd.DataFrame):
    """Generate Global Interpretability framework using SHAP on best Classifer."""
    if not SHAP_AVAILABLE:
        logger.error("Skipping Interpretability: SHAP module is not installed.")
        return

    logger.info(f"--- Generating Interpretability Profiles (SHAP) for {best_model_name} ---")
    model_path = MODELS_DIR / "classification" / f"{best_model_name}.joblib"
    
    if not model_path.exists():
        logger.warning(f"Cannot generate SHAP: Best model '{best_model_name}' missing.")
        return
        
    model = joblib.load(model_path)

    try:
        if "forest" in best_model_name.lower() or "xgb" in best_model_name.lower():
            explainer = shap.TreeExplainer(model)
            # Use smaller background sample for perf
            X_sample = X_eval.sample(min(1000, len(X_eval)), random_state=42)
            shap_values = explainer.shap_values(X_sample)
            
            # Handle binary classifier output structure
            if isinstance(shap_values, list):
                shap_to_plot = shap_values[1]  # positive class
            else:
                shap_to_plot = shap_values

            # Generative Plot
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_to_plot, X_sample, show=False)
            plt.title(f"SHAP Summary (Global Feature Importance) - {best_model_name}")
            plt.tight_layout()
            plt.savefig(INTERP_DIR / f"{best_model_name}_shap_summary.png", dpi=150)
            plt.close()
            logger.info(f"SHAP Explainer saved to {INTERP_DIR}")
            
        else:
            logger.info(f"{best_model_name} is linear. Extrapolating direct weights.")
            weights = pd.Series(model.coef_[0], index=X_eval.columns).sort_values()
            weights.plot(kind="barh", figsize=(10, 8))
            plt.title(f"Feature Weights (Logistic Coefficients) - {best_model_name}")
            plt.tight_layout()
            plt.savefig(INTERP_DIR / f"{best_model_name}_coef_weights.png", dpi=150)
            plt.close()

    except Exception as e:
        logger.error(f"Failed to construct SHAP Interpretations: {e}")


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  4. FAIRNESS AND BIAS DETECTION                                         ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def generate_fairness_report(X_test: pd.DataFrame, y_test: pd.Series, best_model_name: str, class_eval_X: pd.DataFrame):
    """Assess Disparate Impact and Statistical Parity across Demographics."""
    logger.info("--- Extracting AI Governance Fairness Metrics ---")
    
    model = joblib.load(MODELS_DIR / "classification" / f"{best_model_name}.joblib")
    preds = model.predict(class_eval_X)

    df = X_test.copy()
    df["True_Churn"] = y_test.values
    df["Pred_Churn"] = preds
    
    # 1. Reconstruct logical demographics from encoded spaces
    df["Demographic_Age"] = np.where(df["Age"] > 40, "Over 40", "Under 40")
    if "Gender_F" in df.columns and "Gender_M" in df.columns:
        df["Demographic_Gender"] = np.where(df["Gender_F"] == 1, "Female", np.where(df["Gender_M"] == 1, "Male", "Unknown"))
    if "Region_UK" in df.columns and "Region_Europe continentale" in df.columns:
        df["Demographic_Region"] = np.where(df["Region_UK"] == 1, "UK", np.where(df["Region_Europe continentale"] == 1, "Europe", "Other"))

    report_lines = [
        f"# AI Governance & Fairness Report",
        f"**Model Monitored:** {best_model_name}",
        f"**Definition:** Churn=1 (Positive prediction) triggers protective marketing retentions.\n"
    ]
    
    fairness_metrics = {}
    
    for demo_col in [c for c in df.columns if c.startswith("Demographic_")]:
        category_name = demo_col.replace("Demographic_", "")
        groups = df[demo_col].unique()
        if len(groups) < 2: continue
        
        report_lines.append(f"## {category_name} Parity Assessment")
        report_lines.append("| Group | Base Rate P(Y=1) | Selection Rate P(Pred=1) | False Positive Rate |")
        report_lines.append("|-------|------------------|--------------------------|---------------------|")
        
        group_stats = {}
        for g in groups:
            sub = df[df[demo_col] == g]
            if len(sub) == 0: continue
            base_rate = sub["True_Churn"].mean()
            selection_rate = sub["Pred_Churn"].mean()
            fp = ((sub["Pred_Churn"] == 1) & (sub["True_Churn"] == 0)).sum()
            negatives = (sub["True_Churn"] == 0).sum()
            fpr = fp / negatives if negatives > 0 else 0
            
            group_stats[g] = {"selection_rate": selection_rate, "base_rate": base_rate}
            report_lines.append(f"| {g} | {base_rate:.1%} | {selection_rate:.1%} | {fpr:.1%} |")
            
        # Calculate DIR and SPD against the largest group (Assuming Privileged = Max count)
        baseline_group = df[demo_col].value_counts().idxmax()
        report_lines.append(f"\n*Privileged Baseline assumed as majority: {baseline_group}*")
        
        for g in groups:
            if g == baseline_group: continue
            sr_priv = group_stats[baseline_group]["selection_rate"]
            sr_unpriv = group_stats[g]["selection_rate"]
            
            spd = sr_unpriv - sr_priv
            dir_ratio = sr_unpriv / sr_priv if sr_priv > 0 else 0
            
            report_lines.append(f"- **{g} vs {baseline_group}**:")
            report_lines.append(f"  - Statistical Parity Difference (SPD): **{spd:+.3f}** (Ideal: 0)")
            report_lines.append(f"  - Disparate Impact Ratio (DIR): **{dir_ratio:.3f}** (Validation Bounds: 0.8 - 1.25)\n")
            
            if dir_ratio < 0.8 or dir_ratio > 1.25:
                # Add warning syntax to markdown
                report_lines.append(f"> **WARNING:** Found demographic parity violation in {g} vs {baseline_group}.\n")
                
    # Save mathematical JSON
    with open(REPORTS_DIR / "fairness_metrics.json", "w") as f:
        json.dump(fairness_metrics, f, indent=4)
        
    # Document findings formally
    with open(REPORTS_DIR / "fairness_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
        
    logger.info(f"Fairness Profile formally audited and saved to fairness_report.md")
        

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  MAIN ORCHESTRATOR                                                      ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def run_evaluation_pipeline():
    logger.info("="*70)
    logger.info("STARTING PHASE 3 EVALUATION")
    logger.info("="*70)

    # Load Data
    try:
        X_test = pd.read_csv(DATA_DIR / "X_test.csv")
        y_test_clf = pd.read_csv(DATA_DIR / "y_test.csv").squeeze() # Churn Label
    except Exception as e:
        logger.error(f"Failed to load test matrices: {e}")
        return

    # Classification Evaluate
    best_clf, class_eval_X = evaluate_classification_models(X_test, y_test_clf)

    # Regression Evaluate
    if "MonetaryTotal" in X_test.columns:
        evaluate_regression_models(X_test.drop(columns="MonetaryTotal", errors="ignore"), X_test["MonetaryTotal"])

    # Interpretability
    if best_clf:
        generate_shap_explanations(best_clf, class_eval_X)
        generate_fairness_report(X_test, y_test_clf, best_clf, class_eval_X)
        
    logger.info("="*70)
    logger.info("EVALUATION & GOVERNANCE PIPELINE COMPLETE ✓")
    logger.info(f"Outputs saved to: {REPORTS_DIR}")
    logger.info("="*70)


if __name__ == "__main__":
    run_evaluation_pipeline()
