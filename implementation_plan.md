# Technical Backlog: ML Retail Customer Analytics Platform

I'll decompose this comprehensive ML project into a structured, production-ready backlog across 4 logical phases.

---

## **PHASE 1: FOUNDATIONS & INFRASTRUCTURE**

### **Epic 1.1: Development Environment Setup**
- **Task 1.1.1**: Initialize project repository structure
  - Create folder hierarchy (`data/`, `src/`, `models/`, `app/`, `reports/`, `notebooks/`)
  - Initialize Git repository with proper `.gitignore` (exclude `venv/`, `*.pkl`, `data/raw/`, `.env`)
  - Set up branch protection rules (main, develop, feature branches)

- **Task 1.1.2**: Configure Python virtual environment
  - Create venv with Python 3.9+
  - Document activation commands for Windows/Linux/Mac
  - Configure VS Code workspace settings (Python interpreter, linting)

- **Task 1.1.3**: Define project dependencies
  - Create `requirements.txt` with versioned packages:
    - Core: pandas, numpy, scikit-learn, matplotlib, seaborn
    - ML: xgboost, lightgbm, optuna, imbalanced-learn
    - Deployment: flask, flask-cors, gunicorn
    - Utils: joblib, python-dotenv, pydantic
  - Add development dependencies (`requirements-dev.txt`): pytest, black, flake8, jupyter

- **Task 1.1.4**: Create comprehensive README.md
  - Project description and business context
  - Installation instructions (step-by-step)
  - Project structure explanation with visual tree
  - Usage guide for each script
  - Contributing guidelines and code standards

### **Epic 1.2: Data Acquisition & Initial Exploration**
- **Task 1.2.1**: Implement data loading utilities (`src/utils.py`)
  - Function: `load_raw_data()` with error handling
  - Function: `validate_schema()` to check 52 expected features
  - Function: `generate_data_report()` for initial statistics

- **Task 1.2.2**: Create exploratory data analysis notebook
  - EDA notebook: `notebooks/01_initial_exploration.ipynb`
  - Analyze data types, shapes, memory usage
  - Generate summary statistics for all 52 features
  - Document findings on data quality issues

- **Task 1.2.3**: Build data quality assessment module
  - Function: `detect_missing_values()` with visualization
  - Function: `identify_outliers()` using IQR and Z-score methods
  - Function: `check_duplicates()` on CustomerID
  - Generate quality report saved to `reports/data_quality_report.html`

- **Task 1.2.4**: Implement data profiling pipeline
  - Use pandas-profiling or ydata-profiling
  - Generate automated EDA report for all 52 features
  - Document distributions, correlations, warnings
  - Save to `reports/profile_report.html`

---

## **PHASE 2: CORE FEATURES - DATA PREPROCESSING & MODELING**

### **Epic 2.1: Data Cleaning & Transformation**
- **Task 2.1.1**: Build missing value imputation pipeline (`src/preprocessing.py`)
  - Implement `ImputationStrategy` class with methods:
    - `impute_age()`: KNN imputer (k=5) for Age feature
    - `impute_numeric()`: Median for skewed distributions
    - `impute_categorical()`: Mode or "Unknown" constant
  - Save fitted imputers to `models/imputers/`

- **Task 2.1.2**: Develop outlier detection and handling module
  - Function: `handle_support_tickets()` - cap at 99th percentile, replace 999 with median
  - Function: `handle_satisfaction()` - replace -1 and 99 with mode, clip to [0-5]
  - Function: `detect_outliers_isolation_forest()` for multivariate detection
  - Log all transformations for reproducibility

- **Task 2.1.3**: Create date parsing and feature extraction utility
  - Function: `parse_registration_date()`:
    - Handle formats: "DD/MM/YY", "YYYY-MM-DD", "MM/DD/YYYY"
    - Use `pd.to_datetime()` with `dayfirst=True`, `errors='coerce'`
    - Extract: `RegYear`, `RegMonth`, `RegDay`, `RegWeekday`, `RegQuarter`
  - Function: `calculate_account_age_days()` from registration to current date

- **Task 2.1.4**: Implement feature engineering pipeline
  - Create derived features:
    - `MonetaryPerDay = MonetaryTotal / (Recency + 1)`
    - `AvgBasketValue = MonetaryTotal / Frequency`
    - `TenureRatio = Recency / CustomerTenure`
    - `ReturnRate = NegQtyCount / TotalTrans`
    - `CancellationRate = CancelledTrans / TotalTrans`
  - IP feature extraction from `LastLoginIP`:
    - `is_private_ip`, `ip_country` (using GeoIP2 or similar)
  - Save engineered dataset to `data/processed/features_engineered.csv`

- **Task 2.1.5**: Build encoding strategy for categorical features
  - Implement `CategoricalEncoder` class:
    - Ordinal encoding: `AgeCategory`, `SpendingCat`, `LoyaltyLevel`, `BasketSize`, `ChurnRisk`, `PreferredTime` (with proper ordering)
    - One-hot encoding: `CustomerType`, `FavoriteSeason`, `Region`, `WeekendPref`, `ProdDiversity`, `Gender`, `AccountStatus`
    - Target encoding: `Country` (37+ values, high cardinality)
    - Cyclical encoding: `PreferredDay`, `PreferredHour`, `PreferredMonth` (sin/cos transformation)
  - Handle unseen categories in test set
  - Save encoders to `models/encoders/`

- **Task 2.1.6**: Remove redundant and useless features
  - Drop constant feature: `Newsletter` (100% "Yes")
  - Function: `remove_low_variance_features()` with threshold=0.01
  - Function: `remove_correlated_features()`:
    - Calculate correlation matrix
    - Remove features with |correlation| > 0.85
    - Prioritize business-relevant features (keep `MonetaryTotal` over `MonetaryAvg`)
  - Calculate VIF, remove features with VIF > 10

### **Epic 2.2: Feature Scaling & Dimensionality Reduction**
- **Task 2.2.1**: Implement robust scaling pipeline
  - Create `ScalingPipeline` class:
    - StandardScaler for features used in distance-based algorithms
    - RobustScaler for features with outliers
    - MinMaxScaler for neural networks (if applicable)
  - **CRITICAL**: Fit only on `X_train`, transform both train and test
  - Never scale target variable `y` (Churn, MonetaryTotal, etc.)
  - Save fitted scalers to `models/scalers/`

- **Task 2.2.2**: Build PCA dimensionality reduction module
  - Function: `apply_pca()`:
    - Fit PCA on scaled training data
    - Analyze explained variance ratio
    - Select components explaining 85-95% variance
    - Generate scree plot and cumulative variance plot
  - Create 2D/3D visualizations for clustering
  - Save PCA transformer and component analysis to `models/pca/`

- **Task 2.2.3**: Create train/test split utility
  - Function: `create_train_test_split()`:
    - Split ratio: 80% train, 20% test
    - Stratify on target variable (Churn) to preserve class distribution
    - Set `random_state=42` for reproducibility
  - Save splits to `data/train_test/` as CSV files
  - Log split statistics (sizes, class distributions)

### **Epic 2.3: Machine Learning Modeling**
- **Task 2.3.1**: Implement clustering models (`src/train_model.py`)
  - **K-Means Clustering**:
    - Determine optimal k using Elbow method and Silhouette score
    - Fit on PCA-reduced features
    - Analyze cluster characteristics (RFM profiles)
  - **DBSCAN**:
    - Tune eps and min_samples parameters
    - Identify outlier customers (noise points)
  - Save cluster labels and centroids to `models/clustering/`

- **Task 2.3.2**: Build classification pipeline for Churn prediction
  - Implement multiple classifiers:
    - **Logistic Regression** (baseline)
    - **Random Forest** (feature importance analysis)
    - **XGBoost** (high performance)
    - **LightGBM** (fast training)
  - Handle class imbalance using:
    - SMOTE (Synthetic Minority Oversampling)
    - Class weights adjustment
    - Undersampling majority class
  - Save trained models to `models/classification/`

- **Task 2.3.3**: Develop regression models for revenue prediction
  - Target: `MonetaryTotal` or `MonetaryAvg`
  - Implement models:
    - **Linear Regression** (baseline, interpretability)
    - **Ridge/Lasso Regression** (regularization)
    - **Random Forest Regressor**
    - **Gradient Boosting Regressor**
  - Feature importance analysis
  - Save models to `models/regression/`

- **Task 2.3.4**: Create hyperparameter optimization module
  - Implement `GridSearchOptimizer` class:
    - Define parameter grids for each model type
    - Use 5-fold cross-validation
    - Scoring: F1-score (classification), R² (regression)
  - Implement `OptunaOptimizer` class:
    - Define search spaces using Optuna
    - Run 100-200 trials with pruning
    - Compare with GridSearch results
  - Log optimization history to `reports/hyperparameter_tuning/`

### **Epic 2.4: Code Audit & Architecture Refactoring (NEW)**
> [!WARNING]
> Audit uncovered severe Data Leakage architecture bugs in `src/preprocessing.py`.

- **Task 2.4.1**: Refactor `src/preprocessing.py` order of operations
  - Bug: `TargetEncoding` and `KNNImputation` are currently calculating statistics over the *entire dataset* (including the Test Set) before the `train_test_split`.
  - Fix: Move `train_test_split` up in the pipeline script, running it immediately after deterministic date parsing.
  
- **Task 2.4.2**: Decouple `fit` and `transform` logic
  - Refactor `CategoricalEncoder` to include distinct `fit(X, y)` and `transform(X)` methods.
  - Refactor `ImputationStrategy` to include distinct `fit(X)` and `transform(X)` methods.
  - Apply `fit_transform` exclusively to `X_train` while preserving state.
  - Apply isolated `.transform()` on `X_test`.

## User Review Required
The proposed fix requires partially rewriting `src/preprocessing.py` to decouple the stateful transformers and correct the orchestration order. Please review Epic 2.4 and approve this architectural change before we finalize the data structures and proceed to Phase 3.

---

## **PHASE 3: MODEL EVALUATION, SECURITY & AI GOVERNANCE**

### **Epic 3.1: Model Evaluation & Validation**
- **Task 3.1.1**: Build comprehensive evaluation module (`src/evaluate.py`)
  - **Classification metrics**:
    - Confusion matrix, precision, recall, F1-score, ROC-AUC
    - Classification report by class
    - Precision-Recall curve for imbalanced data
  - **Regression metrics**:
    - MAE, MSE, RMSE, R², MAPE
    - Residual plots and distribution analysis
  - **Clustering metrics**:
    - Silhouette score, Davies-Bouldin index, Calinski-Harabasz score
  - Save evaluation reports to `reports/evaluation/`

- **Task 3.1.2**: Implement cross-validation framework
  - Function: `perform_cross_validation()`:
    - Stratified K-Fold (k=5) for classification
    - Regular K-Fold for regression
    - Time-series split if temporal features are critical
  - Calculate mean and std of metrics across folds
  - Detect overfitting (large train-test performance gap)

- **Task 3.1.3**: Create model interpretability module
  - **SHAP values** for model explanations:
    - TreeExplainer for tree-based models
    - Generate waterfall plots for individual predictions
    - Summary plots for feature importance
  - **LIME** for local interpretability
  - **Permutation importance** for all model types
  - Save visualizations to `reports/interpretability/`

- **Task 3.1.4**: Build model comparison dashboard
  - Compare all trained models on common metrics
  - Create leaderboard with sortable columns
  - Visualize performance trade-offs (precision vs recall, speed vs accuracy)
  - Export to `reports/model_comparison.html`

### **Epic 3.2: Data Security & Privacy**
- **Task 3.2.1**: Implement data anonymization utilities
  - Function: `anonymize_customer_id()`: Hash CustomerID using SHA-256
  - Function: `mask_ip_addresses()`: Truncate last octet of LastLoginIP
  - Function: `remove_pii()`: Drop or encrypt sensitive features
  - Apply before sharing data externally

- **Task 3.2.2**: Create data access control layer
  - Implement role-based access control (RBAC):
    - Analyst: Read-only access to processed data
    - Data Scientist: Full access to train models
    - Admin: Access to raw data and production models
  - Log all data access events to `logs/access.log`

- **Task 3.2.3**: Build data validation and schema enforcement
  - Use Pydantic models to enforce data types
  - Validate incoming data against expected schema
  - Reject malformed inputs with clear error messages
  - Log validation failures for monitoring

### **Epic 3.3: AI Governance & Bias Detection**
- **Task 3.3.1**: Implement fairness metrics calculation
  - Analyze model performance across demographic groups:
    - Gender (if ethical and relevant)
    - AgeCategory
    - Region
  - Calculate disparate impact ratio
  - Detect statistical parity violations
  - Document findings in `reports/fairness_report.md`

- **Task 3.3.2**: Create model monitoring framework
  - Track prediction drift over time:
    - Distribution shifts in input features
    - Changes in prediction confidence
    - Performance degradation on new data
  - Implement alerting for anomalies
  - Save monitoring metrics to time-series database or CSV

- **Task 3.3.3**: Build model versioning system
  - Use MLflow or custom versioning:
    - Track model parameters, metrics, artifacts
    - Tag models with version numbers (v1.0, v1.1, etc.)
    - Enable rollback to previous versions
  - Store in `models/versions/` with metadata JSON

---

## **PHASE 4: DEPLOYMENT & PRODUCTION**

### **Epic 4.1: Flask API Development**
- **Task 4.1.1**: Create robust Flask application structure (`app/`)
  - Initial directory scaffolding: 
    - `app/app.py`: Main Flask initialization script.
    - `app/routes.py`: Controller mapping HTTP endpoints to ML models.
    - `app/templates/`: Jinja2 Web UI Frontends.
  - Load joblib models (Classification, Regression, Scalers, PCA) upon server boot to minimize inference latency.

- **Task 4.1.2**: Implement inference prediction endpoints
  - **POST `/api/predict/churn`**:
    - Validate parsed JSON instantly with Phase 3's `CustomerInferencePayload` Pydantic models.
    - Funnel safe payload through `src.preprocessing` transformations.
    - Inject into `logistic_regression.joblib` and return strict `{"churn_probability": 0.xx, "risk_level": "High/Low"}` formats.
  - **POST `/api/predict/revenue`**:
    - Mirror Pydantic security to protect Ridge/RF predictions.
  - Apply global error handling for missing inferences and boundary violations.

- **Task 4.1.3**: Build batch prediction endpoint
  - **POST `/api/predict/batch`**:
    - Accept CSV file upload with multiple customers
    - Process in batches to avoid memory issues
    - Return results as downloadable CSV
  - Implement async processing for large files (Celery/Redis)

- **Task 4.1.4**: Create model management endpoints
  - **GET `/api/models`**: List available models with metadata
  - **POST `/api/models/reload`**: Hot-reload model without restart
  - **GET `/api/health`**: Health check endpoint for monitoring
  - **GET `/api/metrics`**: Return model performance metrics

### **Epic 4.2: Frontend Interface**
- **Task 4.2.1**: Design simple web interface (`app/templates/`)
  - Create HTML forms for single predictions:
    - Input fields for key customer features
    - Submit button to call API
    - Display prediction result with confidence score
  - Use Bootstrap or Tailwind for styling

- **Task 4.2.2**: Build batch prediction interface
  - File upload component for CSV
  - Progress bar for processing status
  - Download button for results
  - Error display for validation failures

- **Task 4.2.3**: Create model insights dashboard
  - Display feature importance charts
  - Show cluster visualizations (PCA 2D plot)
  - Model performance metrics table
  - Use Chart.js or Plotly for interactive visualizations

### **Epic 4.3: Testing & Quality Assurance**
- **Task 4.3.1**: Implement unit tests (`tests/`)
  - Test preprocessing functions (imputation, encoding, scaling)
  - Test model prediction outputs with sample data
  - Test API endpoints with mock data
  - Achieve >80% code coverage

- **Task 4.3.2**: Create integration tests
  - Test end-to-end pipeline: raw data → preprocessing → prediction
  - Test Flask API with real HTTP requests
  - Validate response schemas
  - Test error handling paths

- **Task 4.3.3**: Build data validation tests
  - Test schema enforcement with invalid inputs
  - Test handling of missing values
  - Test handling of out-of-range values
  - Test handling of unseen categorical values



### **Epic 4.5: Documentation & Knowledge Transfer**
- **Task 4.5.1**: Create API documentation
  - Use Swagger/OpenAPI specification
  - Document all endpoints with examples
  - Include authentication requirements
  - Add code samples in Python and curl

- **Task 4.5.2**: Write technical documentation
  - Architecture overview diagram
  - Data pipeline flowchart
  - Model training process documentation
  - Deployment architecture
  - Troubleshooting guide

- **Task 4.5.3**: Create user guide
  - How to use the web interface
  - How to interpret predictions
  - FAQ section
  - Contact information for support

---

## **Backlog Summary**

| Phase | Epics | Tasks | Estimated Effort |
|-------|-------|-------|------------------|
| Phase 1: Foundations | 2 | 8 | 2-3 weeks |
| Phase 2: Core Features | 3 | 17 | 4-6 weeks |
| Phase 3: Security & AI | 3 | 10 | 2-3 weeks |
| Phase 4: Deployment | 5 | 15 | 3-4 weeks |
| **TOTAL** | **13** | **50** | **11-16 weeks** |

This structured backlog provides a production-ready roadmap with clear dependencies, technical specifications, and actionable tasks for each phase of the ML retail analytics platform.