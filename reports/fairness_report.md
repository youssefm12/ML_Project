# AI Governance & Fairness Report
**Model Monitored:** logistic_regression
**Definition:** Churn=1 (Positive prediction) triggers protective marketing retentions.

## Gender Parity Assessment
| Group | Base Rate P(Y=1) | Selection Rate P(Pred=1) | False Positive Rate |
|-------|------------------|--------------------------|---------------------|
| Female | 33.2% | 58.1% | 46.9% |
| Male | 34.4% | 55.2% | 37.9% |
| Unknown | 32.4% | 52.6% | 41.7% |

*Privileged Baseline assumed as majority: Unknown*
- **Female vs Unknown**:
  - Statistical Parity Difference (SPD): **+0.055** (Ideal: 0)
  - Disparate Impact Ratio (DIR): **1.104** (Validation Bounds: 0.8 - 1.25)

- **Male vs Unknown**:
  - Statistical Parity Difference (SPD): **+0.025** (Ideal: 0)
  - Disparate Impact Ratio (DIR): **1.048** (Validation Bounds: 0.8 - 1.25)

## Region Parity Assessment
| Group | Base Rate P(Y=1) | Selection Rate P(Pred=1) | False Positive Rate |
|-------|------------------|--------------------------|---------------------|
| UK | 33.2% | 55.7% | 43.0% |
| Europe | 28.8% | 36.5% | 21.6% |
| Other | 45.5% | 77.3% | 66.7% |

*Privileged Baseline assumed as majority: UK*
- **Europe vs UK**:
  - Statistical Parity Difference (SPD): **-0.191** (Ideal: 0)
  - Disparate Impact Ratio (DIR): **0.656** (Validation Bounds: 0.8 - 1.25)

> **WARNING:** Found demographic parity violation in Europe vs UK.

- **Other vs UK**:
  - Statistical Parity Difference (SPD): **+0.216** (Ideal: 0)
  - Disparate Impact Ratio (DIR): **1.388** (Validation Bounds: 0.8 - 1.25)

> **WARNING:** Found demographic parity violation in Other vs UK.
