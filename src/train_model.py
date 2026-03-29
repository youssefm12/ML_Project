import os
import logging
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import silhouette_score
import optuna
from imblearn.over_sampling import SMOTE
import warnings

try:
    from xgboost import XGBClassifier, XGBRegressor
except ImportError:
    XGBClassifier = None
    XGBRegressor = None

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
except ImportError:
    LGBMClassifier = None
    LGBMRegressor = None

warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("train_model")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"

###############################################################################
# 1. CLUSTERING
###############################################################################
def train_clustering(X_pca: pd.DataFrame, save_dir: Path):
    """Train clustering models on PCA-reduced data."""
    logger.info("--- Starting Clustering ---")
    dest = save_dir / "clustering"
    dest.mkdir(parents=True, exist_ok=True)
    
    # K-Means
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_pca)
    score = silhouette_score(X_pca, clusters)
    logger.info(f"K-Means (k=4) Silhouette Score: {score:.4f}")
    
    joblib.dump(kmeans, dest / "kmeans_model.joblib")
    pd.DataFrame(clusters, columns=['Cluster']).to_csv(dest / "kmeans_labels.csv", index=False)
    
    return kmeans

###############################################################################
# 2. CLASSIFICATION (CHURN)
###############################################################################
def train_classification(X_train: pd.DataFrame, y_train: pd.Series, save_dir: Path):
    """Train classification models for Churn prediction."""
    logger.info("--- Starting Classification ---")
    dest = save_dir / "classification"
    dest.mkdir(parents=True, exist_ok=True)
    
    # -------------------------------------------------------------------------
    # PREVENTION OF DATA LEAKAGE for CHURN
    # The user noted that time/tenure labels (Recency, Season, Month, Time) 
    # were used to synthetically construct 'Churn'. 
    # To achieve a realistic AUC (~0.79), we restrict classification to purely
    # transactional, behavioral, and demographic features.
    # -------------------------------------------------------------------------
    allowed_features = [
        "Frequency", "MonetaryTotal", "MonetaryAvg", "MonetaryStd", 
        "MonetaryMin", "MonetaryMax", "TotalQuantity", 
        "AvgQuantityPerTransaction", "MinQuantity", "MaxQuantity",
        "UniqueProducts", "UniqueDescriptions", "AvgProductsPerTransaction",
        "NegativeQuantityCount", "ZeroPriceCount", "CancelledTransactions",
        "ReturnRatio", "TotalTransactions", "UniqueInvoices", "AvgLinesPerInvoice",
        "Age", "SupportTicketsCount", "SatisfactionScore"
    ]
    
    # Keep only the allowed features that physically exist in X_train
    keep_cols = [c for c in allowed_features if c in X_train.columns]
    X_train_clean = X_train[keep_cols]
    
    logger.info(f"Using {len(keep_cols)} strictly transactional features for Churn Classification.")

    # Handle Imbalance
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train_clean, y_train)
    logger.info(f"SMOTE: Class balance before {y_train.value_counts().to_dict()}, after {pd.Series(y_train_res).value_counts().to_dict()}")
    
    # Baseline Logistic Regression
    lr = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
    lr.fit(X_train_res, y_train_res)
    joblib.dump(lr, dest / "logistic_regression.joblib")
    logger.info("Trained Logistic Regression Baseline.")

    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1)
    rf.fit(X_train_res, y_train_res)
    joblib.dump(rf, dest / "random_forest.joblib")
    logger.info("Trained Random Forest Classifier.")

    # XGBoost
    if XGBClassifier is not None:
        xgb = XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='logloss', random_state=42)
        xgb.fit(X_train_res, y_train_res)
        joblib.dump(xgb, dest / "xgboost.joblib")
        logger.info("Trained XGBoost Classifier.")

    return dest

###############################################################################
# 3. REGRESSION (REVENUE PREDICTION)
###############################################################################
def train_regression(X_train: pd.DataFrame, original_y: pd.Series, save_dir: Path):
    """Train regression models for MonetaryTotal."""
    logger.info("--- Starting Regression ---")
    dest = save_dir / "regression"
    dest.mkdir(parents=True, exist_ok=True)
    
    # Since X_train is scaled, and we want to predict MonetaryTotal.
    # We must ensure we remove any derived Monetary features otherwise extreme data leakage occurs.
    # For now, let's assume original_y is MonetaryTotal and we drop 'MonetaryTotal', 'MonetaryPerDay', 'AvgBasketValue' from X_train
    leak_cols = [c for c in ['MonetaryTotal', 'MonetaryPerDay', 'AvgBasketValue', 'MonetaryAvg', 'MonetaryStd', 'MonetaryMin', 'MonetaryMax'] if c in X_train.columns]
    X_train_clean = X_train.drop(columns=leak_cols)
    
    logger.info(f"Dropped {len(leak_cols)} leak columns for Regression: {leak_cols}")
    
    ridge = Ridge(alpha=1.0, random_state=42)
    ridge.fit(X_train_clean, original_y)
    joblib.dump(ridge, dest / "ridge_regression.joblib")
    logger.info("Trained Ridge Regression Baseline.")
    
    rf_reg = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
    rf_reg.fit(X_train_clean, original_y)
    joblib.dump(rf_reg, dest / "rf_regression.joblib")
    logger.info("Trained Random Forest Regressor.")

    return dest

###############################################################################
# ORCHESTRATOR
###############################################################################
def run_modeling_pipeline():
    logger.info("="*50)
    logger.info("STARTING ML MODELING PIPELINE")
    logger.info("="*50)
    
    train_dir = DATA_DIR / "train_test"
    try:
        X_train = pd.read_csv(train_dir / "X_train.csv")
        y_train = pd.read_csv(train_dir / "y_train.csv").squeeze('columns')
        
        # Load PCA for clustering
        pca_dir = MODELS_DIR / "pca"
        pca = joblib.load(pca_dir / "pca.joblib") 
        X_train_pca = pca.transform(X_train) 
        # Note: actually preprocessing scaled it before PCA. We should read the scaled features or re-scale.
        # But for simplicity since preprocessing saved the pca features, let's just re-calculate from X_train (wait X_train is ALREADY SCALED by preprocessing!)
        X_train_pca = pca.transform(X_train) 
        
    except Exception as e:
        logger.error(f"Failed to load data artifacts: {e}. Please run preprocessing first.")
        return

    # 1. Clustering
    train_clustering(X_train_pca, MODELS_DIR)
    
    # 2. Classification
    train_classification(X_train, y_train, MODELS_DIR)
    
    # 3. Regression
    # We need the true MonetaryTotal from X_train as our regression target.
    if 'MonetaryTotal' in X_train.columns:
        y_reg = X_train['MonetaryTotal']
        train_regression(X_train, y_reg, MODELS_DIR)
    else:
        logger.warning("MonetaryTotal not found in X_train. Skipping Regression models.")

    logger.info("="*50)
    logger.info("ML MODELING PIPELINE COMPLETE")
    logger.info("="*50)

if __name__ == "__main__":
    run_modeling_pipeline()
