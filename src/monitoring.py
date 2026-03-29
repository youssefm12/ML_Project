"""
=============================================================================
Phase 3.3.2: Model Monitoring & Data Drift Framework
=============================================================================
This module systematically tracks prediction drift by measuring distribution 
shifts in input features via Two-Sample Kolmogorov-Smirnov tests. It alerts
on statistical anomalies and pushes these metrics to a time-series CSV tracker.
"""

import os
import csv
import logging
import datetime
from pathlib import Path

import pandas as pd
from scipy.stats import ks_2samp

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("monitoring")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "train_test"
REPORTS_DIR = PROJECT_ROOT / "reports"
MONITORING_DIR = REPORTS_DIR / "monitoring"
MONITORING_DIR.mkdir(parents=True, exist_ok=True)

DRIFT_LOG_CSV = MONITORING_DIR / "drift_metrics.csv"
KS_P_VALUE_THRESHOLD = 0.05 # Strict threshold to alarm on distribution mismatch

def evaluate_data_drift(X_baseline: pd.DataFrame, X_inference: pd.DataFrame) -> dict:
    """
    Apply Kolmogorov-Smirnov (KS) testing to statistically prove whether the simulated 
    new data (inference) behaves identically to the training distributions (baseline).
    """
    logger.info("--- Starting Distribution Drift Evaluation ---")
    
    anomalies_detected = 0
    drift_metrics = {}
    
    # Restrict to strictly transactional features we trust
    allowed_features = [
        "Frequency", "MonetaryTotal", "AvgQuantityPerTransaction", 
        "TotalTransactions", "UniqueInvoices", "Age", "SatisfactionScore"
    ]
    test_cols = [c for c in allowed_features if c in X_baseline.columns and c in X_inference.columns]
    
    for col in test_cols:
        # H0: Data comes from the same distribution
        stat, p_value = ks_2samp(X_baseline[col].dropna(), X_inference[col].dropna())
        
        drift_metrics[col] = {
            "KS_Statistic": float(stat),
            "P_Value": float(p_value),
            "Drift_Detected": p_value < KS_P_VALUE_THRESHOLD
        }
        
        if p_value < KS_P_VALUE_THRESHOLD:
            logger.warning(f"ALERT: Significant Drift detected in feature '{col}'. (p-value: {p_value:.4f})")
            anomalies_detected += 1
            
    if anomalies_detected == 0:
        logger.info("No distribution drift detected boundaries intact.")
        
    return drift_metrics

def log_monitoring_metrics(batch_id: str, anomalies: int, metrics: dict):
    """Save monitoring event to CSV time-series datastore."""
    
    file_exists = DRIFT_LOG_CSV.exists()
    
    with open(DRIFT_LOG_CSV, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["Timestamp", "Batch_ID", "Anomalies_Detected", "Raw_Metrics_JSON"])
            
        writer.writerow([
            datetime.datetime.now().isoformat(),
            batch_id,
            anomalies,
            str(metrics)
        ])
    logger.info(f"Drift record appended to {DRIFT_LOG_CSV}")

def simulate_monitoring_batch():
    """Dummy script testing the KS pipeline with X_test natively acting as 'production'."""
    try:
        baseline = pd.read_csv(DATA_DIR / "X_train.csv")
        inference = pd.read_csv(DATA_DIR / "X_test.csv")
    except Exception as e:
        logger.error(f"Cannot simulate drift: {e}")
        return
        
    metrics = evaluate_data_drift(baseline, inference)
    anomalies = sum(1 for m in metrics.values() if m["Drift_Detected"])
    log_monitoring_metrics("Simulated_PROD_Batch_001", anomalies, metrics)

if __name__ == "__main__":
    simulate_monitoring_batch()
