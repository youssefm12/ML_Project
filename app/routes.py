"""
app/routes.py
=============================================================================
HTTP Controller Layer.
Exposes the AI models as secure RESTful endpoints.
Includes health monitoring, metric reporting, and model management.
"""

import logging
from flask import Blueprint, request, jsonify, render_template, send_file
import pandas as pd
from io import BytesIO

from app.api import get_churn_prediction, get_revenue_forecast, process_batch_csv
from app.models import registry

# Setup
logger = logging.getLogger(__name__)
bp = Blueprint("main", __name__)

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  1. FRONTEND UI ROUTES                                                  ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

@bp.route("/")
def home():
    """Render the Main Interactive Prediction Dashboard."""
    return render_template("index.html")

@bp.route("/batch")
def batch_page():
    """Render the Batch Prediction Upload Interface."""
    return render_template("batch.html")

@bp.route("/insights")
def insights_page():
    """Render the AI Model Governance & Performance Dashboard."""
    return render_template("insights.html")


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  2. ML INFERENCE ENDPOINTS (SINGLE PREDICTIONS)                         ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

@bp.route("/api/predict/churn", methods=["POST"])
def churn_prediction():
    """Calculate Churn Probability for a unique customer payload."""
    payload = request.get_json()
    if not payload:
        return jsonify({"success": False, "error": "Missing JSON input data."}), 400
    
    result = get_churn_prediction(payload)
    status = 200 if result["success"] else 400
    return jsonify(result), status

@bp.route("/api/predict/revenue", methods=["POST"])
def revenue_prediction():
    """Forecast Expected Lifetime Value for a unique customer payload."""
    payload = request.get_json()
    if not payload:
        return jsonify({"success": False, "error": "Missing JSON input data."}), 400
    
    result = get_revenue_forecast(payload)
    status = 200 if result["success"] else 400
    return jsonify(result), status

@bp.route("/api/segment", methods=["POST"])
def segment_customer():
    """Identify customer segment via KMeans clustering."""
    payload = request.get_json()
    if not payload:
        return jsonify({"success": False, "error": "Missing JSON input data."}), 400
    
    from app.api import get_customer_segment
    result = get_customer_segment(payload)
    status = 200 if result["success"] else 400
    return jsonify(result), status


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  3. BATCH PROCESSING ENDPOINTS                                          ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

@bp.route("/api/predict/batch", methods=["POST"])
def batch_prediction():
    """Process high-volume CSV uploads and return formatted risk report."""
    if "file" not in request.files:
        return jsonify({"success": False, "error": "No CSV file uploaded."}), 400
    
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"success": False, "error": "Filename is empty."}), 400
        
    try:
        csv_content = file.read().decode("utf-8")
        processed_df = process_batch_csv(csv_content)
        
        # Return as downloadable Excel-compatible CSV
        output = BytesIO()
        processed_df.to_csv(output, index=False)
        output.seek(0)
        
        return send_file(
            output,
            mimetype="text/csv",
            as_attachment=True,
            download_name="risk_scoring_batch_report.csv"
        )
    except Exception as e:
        logger.error(f"Endpoint batch error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  4. MODEL MANAGEMENT & MONITORING                                       ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

@bp.route("/api/health")
def health_check():
    """Verify inference pipeline uptime and artifact hydration state."""
    return jsonify({
        "status": "healthy",
        "api_version": "v1.0.2",
        "registry_loaded": registry.is_loaded
    }), 200

@bp.route("/api/models")
def list_models():
    """List available in-memory artifacts and their feature specifications."""
    if not registry.is_loaded: registry.load_artifacts()
    
    return jsonify({
        "models": {
            "churn_classifier": "LogisticRegression (v1.0.2)",
            "revenue_regressor": "RandomForest (v1.0.0)",
            "clustering": "KMeans (v1.0.0)"
        },
        "feature_count": {
            "scaler": len(registry.get("scaler_features")),
            "churn": len(registry.get("clf_features")),
            "revenue": len(registry.get("reg_features"))
        }
    }), 200

@bp.route("/api/metrics")
def get_metrics():
    """Retrieve production-grade evaluation metrics (ROC-AUC, MAE)."""
    # Note: In a real system, these would be pulled dynamically from MLFlow
    # For this demo, we return constants from our validation phase.
    return jsonify({
        "classification": {"roc_auc": 0.7702, "f1_score": 0.6106},
        "regression": {"mae": 1.187, "r2": 0.339}
    }), 200
