# Power BI Dashboard — TB Treatment Churn Prediction

## Data Sources

Connect Power BI Desktop to two CSV files:

| Table Name | File | Description |
|---|---|---|
| `ml_dataset` | `data/processed/ml_dataset.csv` | Full patient feature matrix (800 rows, 39 cols) |
| `top10_risk` | `reports/top10_high_risk_patients.csv` | Top 10 highest-risk patients for action |

**Get Data → Text/CSV → Enable "Detect data types"**

---

## Page 1 — Program Overview

### Visuals

| Visual | Fields | Notes |
|---|---|---|
| KPI Card | COUNT(patient_id) | "Total Patients" |
| KPI Card | AVERAGEX(ml_dataset, dropout_label) × 100 | "Dropout Rate %" |
| KPI Card | AVERAGE(risk_score) | "Avg Risk Score" — requires column from model output |
| Donut Chart | outcome, COUNT(patient_id) | Outcome distribution |
| Line Chart | start_year + start_quarter, dropout rate | Trend over time |
| Slicer | region | Filter all visuals |
| Slicer | hiv_status | Filter all visuals |
| Slicer | start_year | Filter all visuals |

### Key DAX Measures

```dax
Dropout Rate % = 
AVERAGEX(ml_dataset, ml_dataset[dropout_label]) * 100

High Risk Count = 
COUNTROWS(FILTER(ml_dataset, ml_dataset[risk_score] >= 0.5))

Program Cost Savings = 
VAR FN_Cost = 5000
VAR FP_Cost = 50
VAR Threshold = 0.20
VAR TotalPatients = COUNTROWS(ml_dataset)
VAR DroppedOut = COUNTROWS(FILTER(ml_dataset, ml_dataset[dropout_label] = 1))
VAR Caught = COUNTROWS(FILTER(ml_dataset, 
    ml_dataset[dropout_label] = 1 && ml_dataset[risk_score] >= Threshold))
VAR FalseAlerts = COUNTROWS(FILTER(ml_dataset,
    ml_dataset[dropout_label] = 0 && ml_dataset[risk_score] >= Threshold))
VAR MissedDropouts = DroppedOut - Caught
VAR OptimizedCost = MissedDropouts * FN_Cost + FalseAlerts * FP_Cost
VAR DefaultCost = DroppedOut * FN_Cost  -- No intervention scenario
RETURN DefaultCost - OptimizedCost
```

---

## Page 2 — Risk Segmentation

### Visuals

| Visual | Fields | Notes |
|---|---|---|
| Filled Map | region, AVERAGE(dropout_label) | Requires custom region lat/lon or shape file |
| Histogram | risk_score (binned) | Distribution of model risk scores |
| Table | patient_id, age, sex, region, distance_km, hiv_status, risk_score | Sortable, top 50 |
| Scatter Plot | X: distance_km, Y: age, Color: dropout_label | Size = risk_score |
| Bar Chart | region, AVERAGE(dropout_label) | Dropout rate by region |

### Conditional Formatting (Table)
- risk_score: Red-Yellow-Green scale (1.0 = red, 0.0 = green)
- dropout_label: Red if = 1, Green if = 0

---

## Page 3 — Intervention Planner

### Visuals

| Visual | Fields | Notes |
|---|---|---|
| Table | rank, age, sex, region, distance_km, hiv_status, alcohol_use, risk_score | From top10_risk table |
| Card | [Program Cost Savings] measure | "Estimated savings vs. no intervention" |
| Card | High Risk Count | "Patients flagged for intervention" |
| Bar Chart | Top LIME features (manual entry or from lime_population figure) | Risk drivers |
| Drill-through page | Individual patient profile | patient_id → demographic + risk details |

### Drill-Through Patient Card
Create a hidden page "Patient Profile" with:
- Patient ID, age, sex, region
- Risk score gauge (0–100%)
- Key risk factors (distance, HIV, alcohol, employment)
- Recommendation text (DAX IF risk_score >= 0.5 THEN "🔴 HIGH RISK — Schedule intervention" ELSE "🟡 MODERATE — Monitor")

---

## Refresh Setup

Since data is CSV-based (not live DB):
1. **Manual refresh:** Re-run Python pipeline → overwrite CSVs → Refresh in Power BI
2. **Scheduled refresh (with Power BI Service):** Publish to Power BI Service → configure gateway → schedule daily refresh after pipeline runs

For live DB integration (PostgreSQL):
- Get Data → Database → PostgreSQL
- Connect to `localhost:5432/tb_program`
- Import view `patient_analytics` directly

---

## Theme

Recommended: Use "Executive" built-in theme.
Primary color: `#E74C3C` (risk red)
Secondary color: `#3498DB` (safe blue)
Background: `#F8F9FA`
