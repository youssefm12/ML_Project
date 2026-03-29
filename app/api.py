"""
app/api.py
=============================================================================
Business Logic and ML Inference Layer.
Bridges the gap between raw HTTP requests and validated ML-ready dataframes.
Implements the feature alignment wrapper for high-end reliability.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List
from io import StringIO

from app.models import registry
from src.security import validate_inference_payload

# Configuration
logger = logging.getLogger(__name__)

def align_and_scale(payload_dict: Dict[str, Any]) -> pd.DataFrame:
    """
    Standardize incoming JSON against the global Scaling feature set.
    Ensures that features missing from the request are defaulted to neutral 0.0.
    """
    # 1. Access in-memory artifacts
    scale_cols = registry.get("scaler_features")
    clf_cols = registry.get("clf_features")
    reg_cols = registry.get("reg_features")
    scaler = registry.get("robust_scaler")
    
    # 2. Re-establish combined feature space (Scale + Predictors)
    combined_cols = list(set(scale_cols.tolist() + clf_cols.tolist() + reg_cols.tolist()))
    
    # 3. Structural alignment
    inference_matrix = pd.DataFrame(0.0, index=[0], columns=combined_cols)
    for k, v in payload_dict.items():
        if k in inference_matrix.columns:
            inference_matrix.at[0, k] = v if v is not None else 0.0
            
    # 4. Scale strictly the numeric subset the Scaler was fitted on
    scaler_input = inference_matrix[scale_cols]
    scaled_vals = scaler.transform(scaler_input)
    
    # 5. Hydrate processed dataframe and merge back pass-through features
    processed_df = pd.DataFrame(scaled_vals, columns=scale_cols, index=[0])
    for col in inference_matrix.columns:
        if col not in processed_df.columns:
            processed_df[col] = inference_matrix[col]
            
    return processed_df


def get_churn_prediction(payload_json: Dict[str, Any]) -> Dict[str, Any]:
    """Execute Churn Classification inference with Pydantic boundary check."""
    try:
        # Pydantic Structural Governance
        safe_payload = validate_inference_payload(payload_json).model_dump()
        
        # Alignment & Scaling
        processed_df = align_and_scale(safe_payload)
        
        # Inference
        model = registry.get("churn_clf")
        clf_features = registry.get("clf_features")
        reg_features = registry.get("reg_features")
        
        prob_churn = float(model.predict_proba(processed_df[clf_features])[0][1])
        pred_rev = float(registry.get("revenue_reg").predict(processed_df[reg_features])[0])
        
        risk_level = "High Risk" if prob_churn > 0.45 else "Low Risk"
        
        return {
            "success": True,
            "churn_probability": round(prob_churn, 4),
            "risk_level": risk_level,
            "predicted_revenue": round(pred_rev, 2)
        }
    except Exception as e:
        # Definitive check for Pydantic V2 validation failures
        if "ValidationError" in str(type(e)):
            return {
                "success": False, 
                "error": "Validation Error", 
                "pydantic_errors": e.errors()
            }
        logger.error(f"Inference failure: {e}")
        return {"success": False, "error": str(e)}


def get_revenue_forecast(payload_json: Dict[str, Any]) -> Dict[str, Any]:
    """Execute Revenue Continuous Regression inference."""
    try:
        safe_payload = validate_inference_payload(payload_json).model_dump()
        processed_df = align_and_scale(safe_payload)
        
        model = registry.get("revenue_reg")
        reg_features = registry.get("reg_features")
        
        pred = float(model.predict(processed_df[reg_features])[0])
        
        return {
            "success": True,
            "predicted_revenue": round(pred, 2)
        }
    except Exception as e:
        if hasattr(e, "errors") and callable(e.errors):
            return {"success": False, "error": "Validation Error", "pydantic_errors": e.errors()}
        logger.error(f"Revenue forecast failure: {e}")
        return {"success": False, "error": str(e)}


def get_customer_segment(payload_json: Dict[str, Any]) -> Dict[str, Any]:
    """Execute KMeans clustering to identify customer behavioral segments."""
    try:
        # Pydantic Structural Governance
        safe_payload = validate_inference_payload(payload_json).model_dump()
        
        # Alignment & Scaling
        processed_df = align_and_scale(safe_payload)
        
        # Inference
        model = registry.get("clustering")
        if not model:
            return {"success": False, "error": "Clustering model not loaded."}
            
        feat_cols = registry.get("clustering_features")
        
        # Ensure all columns exist for clustering
        cluster_input = processed_df.reindex(columns=feat_cols, fill_value=0.0)
        segment = int(model.predict(cluster_input)[0])
        
        # Segment Mapping (Conceptual for Retail)
        mapping = {0: "Low Value", 1: "At-Risk Whale", 2: "Loyal Core", 3: "Recent High-Spender"}
        
        return {
            "success": True,
            "segment_id": segment,
            "segment_label": mapping.get(segment, f"Segment {segment}")
        }
    except Exception as e:
        if hasattr(e, "errors") and callable(e.errors):
            return {
                "success": False, 
                "error": "Validation Error", 
                "pydantic_errors": e.errors()
            }
        logger.error(f"Clustering failure: {e}")
        return {"success": False, "error": str(e)}


def process_batch_csv(csv_content: str) -> pd.DataFrame:
    """
    Process batch CSV customer data for high-volume churn scoring.
    Implements a robust iteration loop over the inference pipeline.
    """
    try:
        df_raw = pd.read_csv(StringIO(csv_content))
        results = []
        
        for index, row in df_raw.iterrows():
            payload = row.to_dict()
            try:
                processed_df = align_and_scale(payload)
                prob = registry.get("churn_clf").predict_proba(processed_df[registry.get("clf_features")])[0][1]
                results.append(round(float(prob), 4))
            except:
                results.append(np.nan)
                
        df_raw["ChurnProbability"] = results
        df_raw["RiskCategory"] = df_raw["ChurnProbability"].apply(
            lambda x: "High Risk" if x > 0.45 else "Low Risk" if pd.notna(x) else "Error"
        )
        return df_raw
    except Exception as e:
        logger.error(f"Batch processing crash: {e}")
        raise
