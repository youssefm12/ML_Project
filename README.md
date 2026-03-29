# Retail Analytics Nexus: Professional Customer Behavioral Forecasting

=============================================================================
*Enterprise-Grade ML Infrastructure for Churn, LTV, and Segmentation.*

This repository contains a full-stack Machine Learning application designed to analyze and predict customer behavior for a large-scale retail gift e-commerce. The project features a hardened pipeline, modular API architecture, and a real-time analytical dashboard.

---

## 🚀 Executive Results Summary (Production Metrics)

| Task                         | Primary Model       | Primary Metric       | Score           | Key Insight                                             |
| :--------------------------- | :------------------ | :------------------- | :-------------- | :------------------------------------------------------ |
| **Churn Forecasting**  | Logistic Regression | **ROC-AUC**    | **0.770** | High recall (0.811) for risk identification.            |
| **Revenue Projection** | Random Forest       | **R² Score**  | **0.650** | Strong correlation for LTV prediction ($MAE = 0.55$). |
| **Segmentation**       | K-Means             | **Silhouette** | **0.62**  | Identified 2 core clusters: Active vs. At-Risk.         |

---

## 🏗️ Technical Architecture

The platform is built using a **Modular Flask App Factory** pattern to ensure zero-latency inference and high maintainability.

- **Unified Model Registry**: Singleton-based loading for all artifacts (Scalers, PCA, XGBoost/RF/Logistic).
- **Structural Pydantic Governance**: 100% of API payloads are validated against Pydantic V2 schemas before ML inference.
- **Feature Alignment Layer**: Robust handling of missing telemetry, ensuring zero crashes even with partial data.

### Project Structure

```text
ML_project/
├── app/                    # Production Flask Application
│   ├── app.py             # App Factory initialization
│   ├── api.py             # Business logic & Model inference
│   ├── models.py          # Unified memory-mapped registry
│   ├── routes.py          # RESTful HTTP controllers
│   ├── templates/         # Glassmorphic Dark-Mode UI
│   └── static/            # CSS/JS Assets
├── src/                    # Pipeline Engineering
│   ├── preprocessing.py    # Hardened cleaning & alignment
│   ├── train_model.py      # Multi-task training script
│   └── security.py         # Pydantic V2 Governance schemas
├── models/                 # Versioned Joblib Artifacts
├── tests/                  # Pytest Comprehensive Suite
├── docs/                   # API Spec & User Guide
└── reports/                # Evaluation & Visualizations
```

---

## 🛠️ Installation & Deployment

### 1. Environment Setup

```powershell
# Create & Activate venv
python -m venv .venv
.venv\Scripts\activate

# Install Dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### 2. Launching the Production API

Always run from the project root using the module flag:

```powershell
$env:PYTHONPATH="."
python -m app.app
```

*Accessible at: `http://localhost:5000`*

---

## 📊 Core Features

- **Single Predict**: Real-time customer profiling with risk-level categorization.
- **Batch Processing**: High-performance CSV processing loop for bulk risk reporting.
- **Model Insights**: Chart.js-powered Performance/Feature Importance visualization.
- **Health Monitoring**: Integrated `/api/health` and `/api/models` metadata endpoints.

---

## 🧪 Testing and Quality

The system is protected by a robust `pytest` suite covering end-to-end integration and preprocessing units.

```powershell
# Run the full test sweep
pytest tests/
```

---

## 👥 Contributors & Support

- **Status**: Production-Ready / Stable
- **Documentation**: See [User Guide](docs/user_guide.md) and [API Specification](docs/api_spec.yaml).
