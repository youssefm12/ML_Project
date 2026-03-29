"""
app/models.py
=============================================================================
Centralized ML Artifact Manager.
Responsible for loading and exposing the trained models, scalers, and encoders.
Ensures zero-latency artifact retrieval by keeping models in memory.
"""

import logging
from pathlib import Path
import joblib

# Setup
logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

class ModelRegistry:
    """Memory-aware singleton for ML model lifecycle management."""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelRegistry, cls).__new__(cls)
            cls._instance.is_loaded = False
            cls._instance.artifacts = {}
        return cls._instance

    def load_artifacts(self):
        """Perform recursive load of all pre-evaluated AI artifacts into RAM."""
        if self.is_loaded:
            return
            
        logger.info("Initializing Enterprise AI Artifact Registry...")
        try:
            # 1. Classification (Logistic Regression)
            self.artifacts["churn_clf"] = joblib.load(MODELS_DIR / "classification" / "logistic_regression.joblib")
            
            # 2. Regression (Random Forest)
            self.artifacts["revenue_reg"] = joblib.load(MODELS_DIR / "regression" / "rf_regression.joblib")
            
            # 3. Clustering (KMeans)
            if (MODELS_DIR / "clustering" / "kmeans_model.joblib").exists():
                self.artifacts["clustering"] = joblib.load(MODELS_DIR / "clustering" / "kmeans_model.joblib")
            
            # 4. Transformers
            self.artifacts["robust_scaler"] = joblib.load(MODELS_DIR / "scalers" / "robust_scaler.joblib")
            self.artifacts["onehot_encoder"] = joblib.load(MODELS_DIR / "encoders" / "onehot_encoder.joblib")
            
            # 5. Feature Indexing (for alignment)
            self.artifacts["scaler_features"] = self.artifacts["robust_scaler"].feature_names_in_
            self.artifacts["clf_features"] = self.artifacts["churn_clf"].feature_names_in_
            self.artifacts["reg_features"] = self.artifacts["revenue_reg"].feature_names_in_
            
            # 6. Clustering Metadata (Dependent on 5)
            if "clustering" in self.artifacts:
                if hasattr(self.artifacts["clustering"], "feature_names_in_"):
                    self.artifacts["clustering_features"] = self.artifacts["clustering"].feature_names_in_
                else:
                    self.artifacts["clustering_features"] = self.artifacts["clf_features"]
            
            self.is_loaded = True
            logger.info("Registry hydrated. Inference Pipeline operational.")
        except Exception as e:
            logger.error(f"CRITICAL: Failed to hydrate artifact registry: {e}")
            raise RuntimeError(f"Artifact Hydration Failure: {e}")

    def get(self, key: str):
        """Retrieve a live artifact from memory."""
        if not self.is_loaded:
            self.load_artifacts()
        return self.artifacts.get(key)

registry = ModelRegistry()
