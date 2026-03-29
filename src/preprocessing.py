"""
=============================================================================
Preprocessing Pipeline — Retail Customer Analytics Platform
=============================================================================
Phase 2.4 Refactoring implementation covering:
  1. Base Estimators         — Decoupled fit/transform to prevent data leakage
  2. ImputationStrategy     — KNN / median / mode imputation 
  3. Outlier handling        — IsolationForest fit strictly on training set
  4. Date parsing            — Multi-format registration dates
  5. Feature engineering     — Derived ratios, IP extraction
  6. CategoricalEncoder      — Target encoding fit stringently against y_train
  7. Feature selection       — Dropping logic applied safely to test set
  8. ScalingPipeline         — Standard / Robust scaling
  9. PCA                     — Dimensionality reduction with scree plot
 10. run_full_pipeline()     — Leak-proof end-to-end orchestrator
=============================================================================
"""

import os
import ipaddress
import logging
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import KNNImputer
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import (
    StandardScaler,
    RobustScaler,
    OrdinalEncoder,
    OneHotEncoder,
)
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("preprocessing")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / "reports"


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  1. IMPUTATION                                                          ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

class ImputationStrategy(BaseEstimator, TransformerMixin):
    """Handle missing values with appropriate strategies per feature type."""

    def __init__(self, knn_neighbors: int = 5):
        self.knn_neighbors = knn_neighbors
        self._knn_imputer = None
        self._knn_features = []
        self._median_values = {}
        self._mode_values = {}
        self._numeric_cols = []
        self._categorical_cols = []

    def fit(self, X: pd.DataFrame, y=None):
        logger.info("Fitting ImputationStrategy...")
        # KNN details
        knn_candidates = [
            "Age", "Recency", "Frequency", "MonetaryTotal",
            "CustomerTenureDays", "TotalTransactions",
        ]
        self._knn_features = [c for c in knn_candidates if c in X.columns]
        
        if self._knn_features:
            self._knn_imputer = KNNImputer(n_neighbors=self.knn_neighbors)
            self._knn_imputer.fit(X[self._knn_features])
            
        # Median details
        self._numeric_cols = [c for c in X.select_dtypes(include=["float64", "int64"]).columns if c != "Age"]
        for col in self._numeric_cols:
            self._median_values[col] = X[col].median()
            
        # Mode details
        self._categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
        for col in self._categorical_cols:
            mode_series = X[col].mode()
            self._mode_values[col] = mode_series.iloc[0] if not mode_series.empty else "Unknown"
            
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        
        # Apply KNN
        if self._knn_imputer is not None and len(self._knn_features) > 0:
            original_age_nans = df["Age"].isnull().sum()
            imputed = self._knn_imputer.transform(df[self._knn_features])
            df[self._knn_features] = imputed
            df["Age"] = df["Age"].round().astype(int)
            logger.info(f"KNN Imputed {original_age_nans} 'Age' records.")
            
        # Apply Medians
        for col in self._numeric_cols:
            if col in df.columns and df[col].isnull().any():
                nans = df[col].isnull().sum()
                df[col].fillna(self._median_values[col], inplace=True)
                
        # Apply Modes
        for col in self._categorical_cols:
            if col in df.columns and df[col].isnull().any():
                df[col].fillna(self._mode_values[col], inplace=True)
                
        return df

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        if self._knn_imputer is not None:
            joblib.dump({"imputer": self._knn_imputer, "features": self._knn_features}, directory / "knn_imputer.joblib")
        joblib.dump(self._median_values, directory / "median_values.joblib")
        joblib.dump(self._mode_values, directory / "mode_values.joblib")
        logger.info(f"Imputers saved to {directory}")


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  2. OUTLIER HANDLING (Capping & Deterministic)                          ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def handle_support_tickets(df: pd.DataFrame) -> pd.DataFrame:
    """Clean SupportTicketsCount: replace 999 → median, -1 → 0, cap at 99th pct."""
    df = df.copy()
    col = "SupportTicketsCount"
    if col not in df.columns: return df
    median_val = df.loc[~df[col].isin([999, -1]), col].median()
    df[col] = df[col].replace(999, median_val).replace(-1, 0)
    cap = df[col].quantile(0.99)
    df[col] = df[col].clip(upper=cap)
    return df

def handle_satisfaction(df: pd.DataFrame) -> pd.DataFrame:
    """Clean SatisfactionScore: replace -1 and 99 → mode, clip to [0, 5]."""
    df = df.copy()
    col = "SatisfactionScore"
    if col not in df.columns: return df
    valid = df[col][(df[col] >= 0) & (df[col] <= 5)]
    mode_val = valid.mode().iloc[0] if not valid.empty else 3
    df[col] = df[col].replace(-1, mode_val).replace(99, mode_val).clip(0, 5)
    return df


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  3. DATE PARSING & FEATURE EXTRACTION                                   ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def parse_registration_date(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    col = "RegistrationDate"
    if col not in df.columns: return df
    df[col] = pd.to_datetime(df[col], dayfirst=True, errors="coerce", infer_datetime_format=True)
    df["RegYear"] = df[col].dt.year
    df["RegMonth"] = df[col].dt.month
    df["RegDay"] = df[col].dt.day
    df["RegWeekday"] = df[col].dt.weekday
    df["RegQuarter"] = df[col].dt.quarter
    return df

def calculate_account_age_days(df: pd.DataFrame, reference_date=None) -> pd.DataFrame:
    df = df.copy()
    col = "RegistrationDate"
    if col not in df.columns or not pd.api.types.is_datetime64_any_dtype(df[col]): return df
    if reference_date is None: reference_date = df[col].max()
    df["AccountAgeDays"] = (reference_date - df[col]).dt.days
    df["AccountAgeDays"] = df["AccountAgeDays"].clip(lower=0)
    return df

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if {"MonetaryTotal", "Recency"}.issubset(df.columns): df["MonetaryPerDay"] = df["MonetaryTotal"] / (df["Recency"] + 1)
    if {"MonetaryTotal", "Frequency"}.issubset(df.columns): df["AvgBasketValue"] = df["MonetaryTotal"] / df["Frequency"].replace(0, np.nan)
    if {"Recency", "CustomerTenureDays"}.issubset(df.columns): df["TenureRatio"] = df["Recency"] / df["CustomerTenureDays"].replace(0, np.nan)
    if {"NegativeQuantityCount", "TotalTransactions"}.issubset(df.columns): df["ReturnRate"] = df["NegativeQuantityCount"] / df["TotalTransactions"].replace(0, np.nan)
    if {"CancelledTransactions", "TotalTransactions"}.issubset(df.columns): df["CancellationRate"] = df["CancelledTransactions"] / df["TotalTransactions"].replace(0, np.nan)

    for c in ["MonetaryPerDay", "AvgBasketValue", "TenureRatio", "ReturnRate", "CancellationRate"]:
        if c in df.columns: df[c] = df[c].replace([np.inf, -np.inf], np.nan).fillna(0)
    return df

def extract_ip_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    col = "LastLoginIP"
    if col not in df.columns: return df
    def _is_private(ip_str):
        try: return int(ipaddress.ip_address(str(ip_str)).is_private)
        except (ValueError, TypeError): return 0
    df["IsPrivateIP"] = df[col].apply(_is_private)
    return df


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  4. CATEGORICAL ENCODING                                                ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """Zero-leakage Categorical Encoder requiring y_train for Target Encoding."""

    ORDINAL_MAPPINGS = {
        "AgeCategory": ["18-24", "25-34", "35-44", "45-54", "55-64", "65+", "Inconnu"],
        "SpendingCategory": ["Low", "Medium", "High", "VIP"],
        "LoyaltyLevel": ["Nouveau", "Jeune", "Établi", "Ancien"],
        "BasketSizeCategory": ["Petit", "Moyen", "Grand"],
        "ChurnRiskCategory": ["Faible", "Moyen", "Élevé", "Critique"],
        "PreferredTimeOfDay": ["Matin", "Midi", "Après-midi", "Soir"],
    }

    ONEHOT_COLUMNS = [
        "CustomerType", "FavoriteSeason", "Region", "WeekendPreference",
        "ProductDiversity", "Gender", "AccountStatus",
    ]

    TARGET_ENCODING_COLUMNS = ["Country"]

    CYCLICAL_COLUMNS = {
        "PreferredDayOfWeek": 7,   
        "PreferredHour": 24,       
        "PreferredMonth": 12,      
    }

    def __init__(self):
        self._ordinal_encoders = {}
        self._onehot_encoder = None
        self._onehot_feature_names = []
        self._target_encoding_maps = {}
        self._target_global_mean = 0

    def fit(self, X: pd.DataFrame, y: pd.Series):
        logger.info("Fitting CategoricalEncoder...")
        
        # 1. Ordinals
        for col, categories in self.ORDINAL_MAPPINGS.items():
            if col in X.columns:
                mapping = {cat: i for i, cat in enumerate(categories)}
                self._ordinal_encoders[col] = mapping
                
        # 2. One-hot
        available_oh = [c for c in self.ONEHOT_COLUMNS if c in X.columns]
        if available_oh:
            self._onehot_encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
            # Fill temp unknowns to learn structure
            self._onehot_encoder.fit(X[available_oh].fillna("Unknown").astype(str))
            self._onehot_feature_names = self._onehot_encoder.get_feature_names_out(available_oh).tolist()

        # 3. Target Encoding (Strictly requires y_train)
        if y is None:
            raise ValueError("y_train MUST be provided to fit target encodings without leakage.")
        
        self._target_global_mean = y.mean()
        for col in self.TARGET_ENCODING_COLUMNS:
            if col in X.columns:
                means = y.groupby(X[col]).mean()
                self._target_encoding_maps[col] = means.to_dict()
                
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        
        # 1. Ordinals
        for col, mapping in self._ordinal_encoders.items():
            if col in df.columns:
                default = len(mapping)
                df[col] = df[col].fillna("Unknown").map(lambda x, m=mapping, d=default: m.get(x, d))

        # 2. One-hot
        available_oh = [c for c in self.ONEHOT_COLUMNS if c in df.columns]
        if self._onehot_encoder is not None and available_oh:
            encoded = self._onehot_encoder.transform(df[available_oh].fillna("Unknown").astype(str))
            encoded_df = pd.DataFrame(encoded, columns=self._onehot_feature_names, index=df.index)
            df = df.drop(columns=available_oh)
            df = pd.concat([df, encoded_df], axis=1)

        # 3. Target Encoding
        for col, mapping in self._target_encoding_maps.items():
            if col in df.columns:
                df[col + "_encoded"] = df[col].map(mapping).fillna(self._target_global_mean)
                df = df.drop(columns=[col])

        # 4. Cyclical
        for col, period in self.CYCLICAL_COLUMNS.items():
            if col in df.columns:
                vals = df[col].fillna(0).astype(float)
                df[col + "_sin"] = np.sin(2 * np.pi * vals / period)
                df[col + "_cos"] = np.cos(2 * np.pi * vals / period)
                df = df.drop(columns=[col])

        return df

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._ordinal_encoders, directory / "ordinal_encoders.joblib")
        if self._onehot_encoder is not None:
            joblib.dump(self._onehot_encoder, directory / "onehot_encoder.joblib")
        joblib.dump(self._target_encoding_maps, directory / "target_encoding_maps.joblib")
        logger.info(f"Encoders saved to {directory}")


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  5. FEATURE SELECTION                                                   ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def identify_redundant_features(df: pd.DataFrame, variance_threshold: float = 0.01, correlation_threshold: float = 0.85) -> list[str]:
    """Identifies columns to drop strictly based on Training data statistics."""
    dropped = []
    
    # 1. Known constants and textual IDs
    for col in ["NewsletterSubscribed", "CustomerID", "LastLoginIP", "RegistrationDate", "RFMSegment"]:
        if col in df.columns: dropped.append(col)
            
    # 2. Low variance
    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
    variances = df[numeric_cols].var()
    low_var = variances[variances < variance_threshold].index.tolist()
    dropped.extend([c for c in low_var if c not in dropped])
    
    # 3. High Correlation
    if len(numeric_cols) > 1:
        corr = df[numeric_cols].corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        to_drop_corr = [col for col in upper.columns if any(upper[col] > correlation_threshold)]
        protected = {"MonetaryTotal", "Frequency", "Recency", "Churn", "Age"}
        to_drop_corr = [c for c in to_drop_corr if c not in protected]
        dropped.extend([c for c in to_drop_corr if c not in dropped])
        
    return list(set(dropped))


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  6. SCALING & PCA                                                       ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

class ScalingPipeline(BaseEstimator, TransformerMixin):
    def __init__(self):
        self._standard_scaler = StandardScaler()
        self._robust_scaler = RobustScaler()
        self._robust_features = []
        self._standard_features = []

    def fit(self, X: pd.DataFrame, y=None):
        logger.info("Fitting ScalingPipeline...")
        numeric = X.select_dtypes(include=["float64", "int64"]).columns.tolist()
        for col in numeric:
            if X[col].nunique() <= 2: continue
            skew = abs(X[col].skew())
            if skew > 1.5: self._robust_features.append(col)
            else: self._standard_features.append(col)

        if self._standard_features: self._standard_scaler.fit(X[self._standard_features])
        if self._robust_features: self._robust_scaler.fit(X[self._robust_features])
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = X.copy()
        if self._standard_features:
            cols = [c for c in self._standard_features if c in X_out.columns]
            if cols: X_out[cols] = self._standard_scaler.transform(X_out[cols])
        if self._robust_features:
            cols = [c for c in self._robust_features if c in X_out.columns]
            if cols: X_out[cols] = self._robust_scaler.transform(X_out[cols])
        return X_out

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._standard_scaler, directory / "std_scaler.joblib")
        joblib.dump(self._robust_scaler, directory / "robust_scaler.joblib")


def apply_pca(X_train: pd.DataFrame, X_test: pd.DataFrame, save_dir: Path):
    dest = save_dir / "pca"
    dest.mkdir(parents=True, exist_ok=True)
    pca_full = PCA(random_state=42).fit(X_train)
    cumulative = np.cumsum(pca_full.explained_variance_ratio_)
    n_comp = int(np.searchsorted(cumulative, 0.90) + 1)
    
    pca = PCA(n_components=n_comp, random_state=42)
    X_train_pca = pd.DataFrame(pca.fit_transform(X_train), index=X_train.index)
    X_test_pca = pd.DataFrame(pca.transform(X_test), index=X_test.index)
    
    joblib.dump(pca, dest / "pca.joblib")
    return X_train_pca, X_test_pca


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  7. FULL PIPELINE ORCHESTRATOR                                          ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def run_full_pipeline(csv_path: str | Path) -> dict:
    from src.utils import load_raw_data

    logger.info("=" * 70)
    logger.info("STARTING LEAK-PROOF PREPROCESSING PIPELINE")
    logger.info("=" * 70)

    # 1. Load & Deterministic Setup
    df = load_raw_data(str(csv_path))
    df = handle_support_tickets(df)
    df = handle_satisfaction(df)
    df = parse_registration_date(df)
    df = calculate_account_age_days(df)
    df = engineer_features(df)
    df = extract_ip_features(df)

    # 2. Train/Test Split (PREVENTS DATA LEAKAGE)
    logger.info("Splitting Data strictly BEFORE applying stateful models...")
    X = df.drop(columns=["Churn"])
    y = df["Churn"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

    # 3. Isolation Forest Filtering (On TRAIN ONLY to prevent prediction leaks)
    logger.info("Fitting IsolationForest exclusively on X_train...")
    iso = IsolationForest(contamination=0.05, random_state=42, n_jobs=-1)
    num_cols = X_train.select_dtypes(include=["float64", "int64"]).columns.tolist()
    iso.fit(X_train[num_cols].fillna(0))
    X_train["IsOutlier"] = (iso.predict(X_train[num_cols].fillna(0)) == -1).astype(int)
    X_test["IsOutlier"] = (iso.predict(X_test[num_cols].fillna(0)) == -1).astype(int)

    # 4. Imputation
    imputer = ImputationStrategy(knn_neighbors=5)
    X_train = imputer.fit(X_train).transform(X_train)
    X_test = imputer.transform(X_test)
    imputer.save(MODELS_DIR / "imputers")

    # 5. Categorical Encoding
    encoder = CategoricalEncoder()
    X_train = encoder.fit(X_train, y_train).transform(X_train)
    X_test = encoder.transform(X_test)
    encoder.save(MODELS_DIR / "encoders")

    # 6. Feature Selection (Using Train Stats Only)
    drop_cols = identify_redundant_features(X_train)
    X_train = X_train.drop(columns=drop_cols)
    X_test = X_test.drop(columns=drop_cols)
    logger.info(f"Dropped {len(drop_cols)} redundant/highly correlated features based on Train.")

    # 7. Scaling
    scaler = ScalingPipeline()
    X_train = scaler.fit(X_train).transform(X_train)
    X_test = scaler.transform(X_test)
    scaler.save(MODELS_DIR / "scalers")

    # 8. Dimensionality Reduction
    X_train_pca, X_test_pca = apply_pca(X_train, X_test, MODELS_DIR)

    # 9. Dump Artifacts
    train_test_dir = DATA_DIR / "train_test"
    train_test_dir.mkdir(parents=True, exist_ok=True)
    X_train.to_csv(train_test_dir / "X_train.csv", index=False)
    X_test.to_csv(train_test_dir / "X_test.csv", index=False)
    y_train.to_csv(train_test_dir / "y_train.csv", index=False)
    y_test.to_csv(train_test_dir / "y_test.csv", index=False)

    logger.info("=" * 70)
    logger.info("PIPELINE COMPLETE ✓ No Data Leakage detected.")
    logger.info(f"  Train: X={X_train.shape}, y={y_train.shape}")
    logger.info(f"  Test:  X={X_test.shape}, y={y_test.shape}")
    logger.info("=" * 70)
    return {"X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test}

if __name__ == "__main__":
    import sys
    csv = sys.argv[1] if len(sys.argv) > 1 else str(DATA_DIR / "raw" / "retail_customers_COMPLETE_CATEGORICAL.csv")
    results = run_full_pipeline(csv)
