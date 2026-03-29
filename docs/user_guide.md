# RetainAI Nexus: User Guide
=============================================================================
*Professional Customer intelligence & Retention Forecasting Platform*

Welcome to the **RetainAI Nexus** user guide. This document provides everything you need to know to leverage our ML-powered platform for customer retention and revenue optimization.

## 1. Getting Started

The platform is accessible via your local deployment at `http://localhost:5000`.

### Navigation
Use the header navigation to switch between the three core modules:
- **Single Predict**: manual entry for individual customer analysis.
- **Batch Process**: high-volume scoring via CSV upload.
- **Model Insights**: real-time visualization of model health and feature weights.

---

## 2. Using the Web Interface

### 2.1 Single Predict (Real-Time Inference)
This module is designed for CSRs (Customer Service Representatives) or Account Managers to check a customer's risk profile during or after an interaction.

1.  **Input Data**: Enter the customer's transactional telemetry.
    - **Frequency**: Total number of historical purchases.
    - **Capital Spent ($LTV)**: Total lifetime value spent.
    - **Avg Units/Cart**: Average number of items per transaction.
    - **Customer Age**: For demographic risk weighting.
2.  **Initiate Forecast**: Click the button to send the payload to the ML engine.
3.  **Interpret Results**:
    - **Churn Probability**: A score from 0-100%. Higher means more likely to leave.
    - **Risk Category**: Automated classification (Low Risk / High Risk).
    - **LTV Delta**: The predicted marginal revenue for the next 30 days.

### 2.2 Batch Process (Bulk Scoring)
Designed for Marketing Analysts to run weekly risk reports.

1.  **Upload CSV**: Drag and drop your customer list.
    - *Note: Ensure your CSV follows the schema in `docs/api_spec.yaml`.*
2.  **Process**: The system will iterate through thousands of rows using our optimized batch pipeline.
3.  **Download Results**: Receive a processed CSV with `Churn_Probability` and `Risk_Label` columns added.

### 2.3 Model Insights
Monitor the "brain" of the operation.

- **Global Performance**: View ROC-AUC and MAE metrics for the production models.
- **Feature Importance**: Understand which telemetry points (e.g., Recency vs. Frequency) are currently driving the most influence in the model's decisions.

---

## 3. How to Interpret Predictions

| Risk Level | Meaning | Recommended Action |
| :--- | :--- | :--- |
| **Low Risk** | Loyal/Stable customer ($<15\%$ churn). | Standard maintenance / Loyalty rewards. |
| **Moderate** | Showing early signs of disengagement. | Personalized newsletter or subtle perk. |
| **High Risk** | High probability of churn ($>45\%$). | **Immediate Outreach**: Discount code or direct contact. |

---

## 4. FAQ

**Q: Why did the prediction fail for a specific customer?**
A: Most likely due to missing "Required" fields (Frequency, MonetaryTotal). Ensure all telemetry points are non-null.

**Q: How often is the model updated?**
A: The current production model is static. Retraining is recommended monthly or after significant promotional events.

**Q: Can I integrate this into our CRM?**
A: Yes. Use the REST API endpoints documented in `docs/api_spec.yaml`.

---

## 5. Support & Contact

For technical failures or architectural queries:
- **Technical Lead**: Antigravity AI (AI Tech Lead)
- **Email**: support@retail-ml-nexus.internal
- **Emergency**: Check the `logs/` directory for runtime stack traces.
