"""
=============================================================================
Phase 4.1: Flask API Engine
=============================================================================
Hosts our fully evaluated ML models directly inside server RAM.
Binds to the premium `templates/index.html` UI and natively routes JSON payloads 
through the Phase 3 Pydantic validation architecture to prevent boundary injection.
"""

import os
import sys
import logging
from pathlib import Path

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib

# Extend sys.path so we can natively import from `src.`
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

try:
    from src.security import validate_inference_payload
    from pydantic import ValidationError
except ImportError as e:
    raise RuntimeError(f"Could not secure API: {e}. Check PYTHONPATH.")


# ---------------------------------------------------------------------------
# Load Pre-Evaluated AI Artifacts (In-Memory Singleton)
# ---------------------------------------------------------------------------
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_LIFECYCLE = {}

def load_ai_artifacts():
    print("[INIT] Loading High-Performance AI Pipeline Artifacts into RAM...")
    try:
        # Load Predictors
        MODEL_LIFECYCLE["churn_clf"] = joblib.load(MODELS_DIR / "classification" / "logistic_regression.joblib")
        MODEL_LIFECYCLE["revenue_reg"] = joblib.load(MODELS_DIR / "regression" / "rf_regression.joblib")
        
        # Load Scale Operators
        MODEL_LIFECYCLE["robust_scaler"] = joblib.load(MODELS_DIR / "scalers" / "robust_scaler.joblib")
        MODEL_LIFECYCLE["std_scaler"] = joblib.load(MODELS_DIR / "scalers" / "std_scaler.joblib")
        
        # Memorize the exact Column structure the UI depends on
        MODEL_LIFECYCLE["scale_features"] = MODEL_LIFECYCLE["robust_scaler"].feature_names_in_
        MODEL_LIFECYCLE["clf_features"] = MODEL_LIFECYCLE["churn_clf"].feature_names_in_
        MODEL_LIFECYCLE["reg_features"] = MODEL_LIFECYCLE["revenue_reg"].feature_names_in_
        print("[INIT] Loaded successfully. Inference Pipeline armed.")
    except Exception as e:
        print(f"[CRITICAL] Artifact load failed: {e}")

load_ai_artifacts()


# ---------------------------------------------------------------------------
# Flask Factory
# ---------------------------------------------------------------------------
app = Flask(__name__)
CORS(app)  # Explicitly enable Cross-Origin handling for the API UI
app.logger.setLevel(logging.INFO)


@app.route("/")
def dashboard():
    """Render the ultra-premium Vue/React-like Single Page Dashboards."""
    return render_template("index.html")


@app.route("/api/predict/churn", methods=["POST"])
def predict_churn():
    """Main production endpoint for calculating realtime Risk Profiling."""
    try:
        raw_json = request.get_json()
        if not raw_json:
            return jsonify({"detail": "Missing JSON payload."}), 400
            
        # 1. Pipeline Gatekeeper: Apply Pydantic boundary schema rules.
        try:
            # Use model_dump() for Pydantic V2 compatibility
            safe_payload = validate_inference_payload(raw_json).model_dump()
        except ValidationError as e:
            # Mask internal error traces, project clean dictionary parsing
            return jsonify({"detail": "Invalid structural payload.", "pydantic_errors": e.errors()}), 400

        # 2. Reconstruct DataFrame from validated payload
        df_raw = pd.DataFrame([safe_payload])
        
        # 3. Professional Feature Alignment Layer
        # The model might expect features that the scaler didn't see (e.g. if they were dropped as redundant).
        # We ensure all features required by BOTH are accounted for.
        scale_cols = MODEL_LIFECYCLE["scale_features"].tolist()
        clf_cols = MODEL_LIFECYCLE["clf_features"].tolist()
        reg_cols = MODEL_LIFECYCLE["reg_features"].tolist()
        
        required_features = list(set(scale_cols + clf_cols + reg_cols))
        
        # Build matrix with defaults for any missing features
        inference_matrix = pd.DataFrame(0.0, index=df_raw.index, columns=required_features)
        for col in df_raw.columns:
            if col in inference_matrix.columns:
                inference_matrix[col] = df_raw[col]

        # 4. Apply Scaling strictly to the subset the Scaler was fitted on
        scaler_input = inference_matrix[scale_cols]
        scaled_vals = MODEL_LIFECYCLE["robust_scaler"].transform(scaler_input)
        
        # Create result dataframe and merge back non-scaled features
        processed_df = pd.DataFrame(scaled_vals, columns=scale_cols, index=df_raw.index)
        for col in inference_matrix.columns:
            if col not in processed_df.columns:
                processed_df[col] = inference_matrix[col]

        # 5. Extract exactly what each Predictor expects
        clf_matrix = processed_df[clf_cols]
        reg_matrix = processed_df[reg_cols]
        
        # 6. Inference Compute
        prob_churn = float(MODEL_LIFECYCLE["churn_clf"].predict_proba(clf_matrix)[0][1])
        pred_rev = float(MODEL_LIFECYCLE["revenue_reg"].predict(reg_matrix)[0])
        
        # 7. Response Packing
        risk_level = "High Risk" if prob_churn > 0.45 else "Low Risk" 
        
        return jsonify({
            "churn_probability": prob_churn,
            "risk_level": risk_level,
            "predicted_revenue": pred_rev
        }), 200

    except Exception as e:
        app.logger.error(f"Inference Crash: {e}")
        return jsonify({"detail": "Internal inference error.", "msg": str(e)}), 500


if __name__ == "__main__":
    # Explicit interface binding
    app.run(host="0.0.0.0", port=5000, debug=False)
