"""
=============================================================================
Phase 3.3.3: Model Versioning System
=============================================================================
This module creates a custom MLFlow-lite registry for tracking hyperparameters,
evaluation metrics, and the serialized model objects in a structured, queryable layout.
"""

import os
import json
import logging
import datetime
from pathlib import Path

import joblib

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("versioning")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VERSIONS_DIR = PROJECT_ROOT / "models" / "versions"
VERSIONS_DIR.mkdir(parents=True, exist_ok=True)

def tag_model_version(model, model_name: str, metrics: dict, params: dict, version_str: str = None) -> Path:
    """Save model, its metadata (metrics, hyperparams), and auto-version it."""
    
    # Auto-increment logic
    if version_str is None:
        existing_versions = [d.name for d in VERSIONS_DIR.iterdir() if d.is_dir() and d.name.startswith("v")]
        if not existing_versions:
            version_str = "v1.0"
        else:
            sorted_versions = sorted([float(v[1:]) for v in existing_versions])
            next_v = sorted_versions[-1] + 1.0
            version_str = f"v{next_v:.1f}"

    version_path = VERSIONS_DIR / version_str
    version_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Tagging and archiving model '{model_name}' under version '{version_str}'")

    # Serialize object
    model_dst = version_path / f"{model_name}.joblib"
    joblib.dump(model, model_dst)
    
    # Store MLFlow-styled tracker metadata
    metadata = {
        "version": version_str,
        "model_name": model_name,
        "timestamp": datetime.datetime.now().isoformat(),
        "parameters": params,
        "metrics": metrics
    }
    
    with open(version_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)
        
    logger.info(f"Version {version_str} finalized at {version_path}")
    return version_path

def rollback_to_version(version_str: str, model_name: str):
    """Load a specific version for rollback operations."""
    target_path = VERSIONS_DIR / version_str / f"{model_name}.joblib"
    if not target_path.exists():
        logger.error(f"Version {version_str} for model {model_name} does not exist.")
        raise FileNotFoundError(f"Missing rollback checkpoint: {target_path}")
        
    logger.info(f"Successfully rolled back to {version_str}")
    return joblib.load(target_path)
