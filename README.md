# 🫁 TB Treatment Churn Prediction: Preventing Treatment Abandonment with ML

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/ML-scikit--learn-orange.svg)](https://scikit-learn.org/)
[![Power BI](https://img.shields.io/badge/BI-Power_BI-F2C811.svg)](https://powerbi.microsoft.com/)

---

## 🔬 Project Overview

Tuberculosis (TB) remains the world's leading cause of death from a single infectious agent (1.6M deaths/year, WHO 2023). A critical challenge in TB control programs is **treatment abandonment (defaulting)**: patients who stop taking their medication for ≥2 consecutive months.

> **Why abandonment matters:**
> - Incomplete treatment causes **relapse** and worsens disease
> - Most critically, it drives **drug resistance** — MDR-TB costs ~150X for drug-sensitive TB
> - Early identification of at-risk patients enables **preventive intervention** at a fraction of the re-treatment cost

This project builds a **churn prediction pipeline** — adapting a well-known business analytics problem to the public health domain — to identify high-risk patients **at enrollment**, before treatment starts.

**Methodological design note:** All predictive features are drawn exclusively from enrollment data (demographics, clinical baseline, social support). This enforces strict temporal validity: the model simulates a real-world early warning system with zero data leakage.

---

## 🎯 Business Problem → Data Science applied in Clinical Context 

| Business Frame | Technical Frame |
|---|---|
| "Which patients will stop their treatment?" | Binary classification: `dropout_label` (0/1) |
| "Why is this patient at risk?" | LIME local explanations per patient |
| "Where should we concentrate resources?" | Regional analysis + feature importance |
| "How much does early intervention save?" | Cost-sensitive threshold optimization |
| "How do we present this to program managers?" | Power BI dashboards |

---

## 📊 Results at a Glance

| Model | AUC-ROC | Avg Precision | CV AUC |
|---|---|---|---|
| Logistic Regression | **0.702** | 0.395 | 0.627 ± 0.020 |
| Random Forest | 0.674 | 0.389 | 0.655 ± 0.039 |
| Gradient Boosting | 0.679 | 0.422 | 0.607 ± 0.041 |

**Top predictors at enrollment:** distance to clinic, patient age, social support type.

**Cost-Sensitive Impact:** Optimizing the classification threshold (FN=$5,000 vs FP=$50) yields estimated savings of **~$475,000 per 1,000 patients** compared to no intervention — translating ML metrics directly into program impact.

---

## 📈 Visual Pipeline

### Data Quality & Distributions
![Missing values heatmap](figures/00_missing_values_heatmap.png)
![Outcomes distribution](figures/01_outcomes_distribution.png)

### Exploratory Analysis
![Regional analysis](figures/02_regional_analysis.png)
![Temporal analysis](figures/03_temporal_analysis.png)
![Risk factors boxplots](figures/04_risk_factors_boxplots.png)
![Categorical dropout rates](figures/05_categorical_dropout_rates.png)

### Survival & Correlation
![Correlation matrix](figures/06_correlation_matrix.png)
![Kaplan-Meier retention curve](figures/07_kaplan_meier.png)

### ML Model Performance
![ROC and Precision-Recall curves](figures/08_roc_pr_curves.png)
![Cost threshold + Feature importance](figures/09_cost_features.png)

### Explainability (LIME)
![LIME individual explanations — top 3 patients](figures/10_lime_top3.png)
![LIME population-level risk drivers](figures/11_lime_population.png)

### Expert-Level Validation
![Model calibration — Brier score + Hosmer-Lemeshow](figures/12_calibration.png)
![Cost sensitivity heatmap — LMIC vs HIC parameter space](figures/13_cost_sensitivity.png)
![Cross-cohort external validation — Latin America vs SSA proxy](figures/14_external_validation.png)

---

## 🚀 Key Features

- **Anti-Leakage Feature Engineering**: All features derived exclusively from enrollment data — no visit-derived behavioral features. Temporal validity enforced by design.
- **3-Model Comparison**: Logistic Regression (baseline), Random Forest, Gradient Boosting — with 5-fold stratified CV.
- **Cost-Sensitive Learning**: Threshold tuned to asymmetric misclassification costs (FN = $5,000 vs FP = $50).
- **LIME Explainability**: Per-patient local explanations + population-level risk driver frequency analysis.
- **Kaplan-Meier Survival Analysis**: Treatment retention curves by region and risk group (manual implementation).
- **Streamlit Dashboard**: 4-tab interactive tool with KPIs, regional maps, top-10 risk list, and individual prediction.
- **Power BI Integration**: Executive-level BI dashboard connected to the processed data layer.

---

## 📁 Project Structure

```
tb-churn-prediction/
│
├── data/
│   ├── generate_tb_data.py           ← Realistic synthetic dataset (WHO-calibrated)
│   ├── raw/                          ← patients.csv, visits.csv, results.csv, treatments.csv
│   ├── processed/                    ← ml_dataset.csv (features + target)
│   └── sql/
│       ├── 01_schema.sql             ← PostgreSQL schema (4 tables)
│       └── 02_analytical_queries.sql ← Analytical VIEWs + master query
│
├── notebooks/
│   ├── 01_data_preparation.ipynb     ← ETL + Feature Engineering + QC audit
│   ├── 02_eda.ipynb                  ← EDA + Kaplan-Meier + Regional analysis
│   ├── 03_ml_modeling.ipynb          ← 3-model comparison + Cost-sensitive threshold
│   └── 04_explainability_lime.ipynb  ← LIME individual + population explainability
│
├── src/
│   └── feature_engineering.py        ← Module: SQL JOIN → ML-ready features
│
├── app/
│   └── dashboard_app.py              ← Streamlit dashboard (4 tabs)
│
├── models/
│   ├── lr_baseline.pkl               ← Logistic Regression (best AUC)
│   ├── rf_tb_model.pkl               ← Random Forest
│   ├── gb_model.pkl                  ← Gradient Boosting
│   ├── encoders.pkl                  ← LabelEncoders for categorical features
│   ├── feature_names_enrollment.pkl  ← Feature order (enrollment-only set)
│   ├── optimal_threshold.pkl         ← Cost-optimized classification threshold
│   └── scaler.pkl                    ← StandardScaler (for LR)
│
├── reports/
│   └── top10_high_risk_patients.csv  ← Actionable output for health workers
│
├── figures/                          ← All generated plots (12 figures)
├── powerbi/
│   └── TB_Churn_Dashboard.md         ← Power BI dashboard documentation
├── requirements.txt
└── README.md
```

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| **Language** | Python 3.9+ |
| **Database** | PostgreSQL (schema + analytical views) |
| **Data** | Pandas, NumPy, SciPy |
| **Visualization** | Matplotlib, Seaborn, Plotly Express |
| **ML** | Scikit-learn (LR, RandomForest, HistGradientBoosting) |
| **Survival Analysis** | Kaplan-Meier (manual implementation) |
| **Explainability** | LIME (lime-tabular) |
| **Deployment** | Streamlit |
| **BI Reporting** | Power BI Desktop |

---

## 📋 Pipeline Workflow

```
4 SQL Tables (patients + visits + results + treatments)
              │
              ▼
┌──────────────────────────────┐
│  1. ETL + Feature Eng.       │  Notebook 01
│  SQL → Pandas                │  · JOIN 4 tables (VIEW patient_analytics)
│                              │  · 27 enrollment features
│                              │  · QC audit (missing, distributions)
└───────────┬──────────────────┘
            │
            ▼
┌──────────────────────────────┐
│  2. EDA                      │  Notebook 02
│  Seaborn + Plotly            │  · Dropout rate by region, sex, HIV
│                              │  · Temporal analysis (month of dropout)
│                              │  · Kaplan-Meier retention curves
│                              │  · Risk factor analysis (Mann-Whitney)
└───────────┬──────────────────┘
            │
            ▼
┌──────────────────────────────┐
│  3. ML Modeling              │  Notebook 03
│  3-Model Comparison          │  · LR + RF + Gradient Boosting
│                              │  · 5-fold stratified CV
│                              │  · Cost-sensitive threshold (FN≫FP)
│                              │  · AUC → USD savings translation
└───────────┬──────────────────┘
            │
            ▼
┌──────────────────────────────┐
│  4. Explainability           │  Notebook 04
│  LIME                        │  · Per-patient local explanations
│                              │  · Top 10 at-risk patients identified
│                              │  · Population-level risk driver frequency
└───────────┬──────────────────┘
            │
        ┌───┴───┐
        ▼       ▼
┌────────────┐ ┌─────────────┐
│ Streamlit  │ │  Power BI   │
│ Dashboard  │ │  Dashboard  │
│ (4 tabs)   │ │ (3 pages)   │
└────────────┘ └─────────────┘
```

---

## 📊 Power BI Dashboard

The Power BI dashboard connects directly to the processed data layer (`data/processed/ml_dataset.csv` + `reports/top10_high_risk_patients.csv`). Three report pages:

**Page 1 — Program Overview**
- KPI cards: total patients, dropout rate (%), treatment success rate, avg risk score
- Dropout trend by month of enrollment (line chart)
- Outcome distribution donut chart
- Slicers: year, region, HIV status

**Page 2 — Risk Segmentation**
- Map visual: dropout rate by region (filled map)
- Risk score distribution histogram (RF model output)
- High-risk patients table (top 50, sortable)
- Scatter: distance vs. age, colored by dropout

**Page 3 — Intervention Planner**
- Top 10 actionable patients (from `reports/top10_high_risk_patients.csv`)
- Cost savings calculator (DAX: threshold × FN/FP cost parameters)
- LIME factor summary: most frequent risk drivers (bar chart)
- Drill-through: individual patient profile card

**Data connection:** Power BI Desktop → Get Data → Text/CSV → point to `data/processed/ml_dataset.csv`

---

## 🚀 Getting Started

```bash
# 1. Clone and setup
git clone https://github.com/facuccini/tb-churn-prediction.git
cd tb-churn-prediction
pip install -r requirements.txt

# 2. Generate synthetic data (WHO-calibrated parameters)
cd data && python generate_tb_data.py && cd ..

# 3. Run notebooks in order
jupyter notebook notebooks/01_data_preparation.ipynb

# 4. Launch Streamlit dashboard
streamlit run app/dashboard_app.py
```
## 🔬 Domain Notes

### WHO Classification of TB Treatment Outcomes
- **Cured**: Bacteriologically confirmed (negative smear/culture at end)
- **Treatment Completed**: Completed without bacteriological confirmation
- **Defaulted**: Treatment interrupted for ≥2 consecutive months ← *our target*
- **Treatment Failed**: Positive smear/culture at month 5+
- **Died**: Any cause during treatment

### On Synthetic Data
This project uses synthetic data calibrated to WHO epidemiological parameters (dropout rate 15–20%, regional variation, evidence-based risk factor odds ratios). Real patient-level datasets (TB Portals, WHO TB-IPD) are available under Data Use Agreements for validated research applications.

### Cost-Sensitive Learning Rationale
- **FN (missed abandonment):** ~$5,000 USD — re-treatment cost + resistance risk
- **FP (unnecessary alert):** ~$50 USD — extra social worker visit
- Threshold is tuned to minimize total program cost, not accuracy

---

## 📚 References

1. WHO Global Tuberculosis Report 2025.
2. Zhang, F. "Using Machine Learning Methods to Predict Early Treatment Outcomes for Multidrug-Resistant or Rifampicin-Resistant Tuberculosis to Enhance Patient Cure Rates: Development and Validation of Multiple Models".
3. Wang L. "A multi-stage machine learning framework for stepwise prediction of tuberculosis treatment outcomes: Integrating gradient boosted decision trees and feature-level analysis for clinical decision support".
4. Chen, J."Predictive machine learning models for anticipating loss to follow-up in tuberculosis patients throughout anti-TB treatment journey".


---

## 👤 About

**Facundo Colaccini, PhD** — Biological Sciences (TB / Drug Discovery) → Data Science

- 🔗 [LinkedIn](www.linkedin.com/in/facundo-colaccini)
- 💻 [GitHub](https://github.com/facuccini)
- 🔬 [See also: Project A — FasR Drug Discovery Pipeline](../fasr-drug-discovery/)

---


---

# 🫁 Predicción de Abandono de Tratamiento TB con ML

> **Versión en español**

---

## 🔬 Resumen del Proyecto

La tuberculosis (TB) sigue siendo la principal causa de muerte por un único agente infeccioso a nivel mundial (1,6 millones de muertes/año, OMS 2023). Uno de los mayores desafíos en los programas de control de TB es el **abandono del tratamiento**: pacientes que interrumpen la medicación por ≥2 meses consecutivos.

> **Por qué importa el abandono:**
> - El tratamiento incompleto genera recaídas y empeora la enfermedad
> - Lo más crítico: impulsa la **resistencia a fármacos** — la TB-MDR cuesta ~$150.000 por caso vs. $1.000 para TB sensible
> - Identificar pacientes en riesgo temprano permite intervenciones preventivas a una fracción del costo del retratamiento

Este proyecto construye un **pipeline de predicción de churn** — adaptando un problema clásico de analytics de negocios al dominio de salud pública — para identificar pacientes de alto riesgo **al momento de la inscripción**, antes de que comience el tratamiento.

**Nota metodológica:** Todas las features predictivas provienen exclusivamente de datos de inscripción (demografía, clínica basal, soporte social). Esto garantiza validez temporal estricta: el modelo simula un sistema de alerta temprana real sin data leakage.

---

## 🎯 Problema de Negocio → Solución Data Science

| Marco de Negocio | Marco Técnico |
|---|---|
| "¿Qué pacientes van a abandonar el tratamiento?" | Clasificación binaria: `dropout_label` (0/1) |
| "¿Por qué está en riesgo este paciente?" | Explicaciones locales LIME por paciente |
| "¿Dónde concentrar los recursos?" | Análisis regional + importancia de features |
| "¿Cuánto ahorra la intervención temprana?" | Optimización de umbral costo-sensible |
| "¿Cómo presentamos esto a los gestores del programa?" | Dashboards Streamlit + Power BI |

---

## 📊 Resultados Principales

| Modelo | AUC-ROC | Precisión Promedio | CV AUC |
|---|---|---|---|
| Regresión Logística | **0.702** | 0.395 | 0.627 ± 0.020 |
| Random Forest | 0.674 | 0.389 | 0.655 ± 0.039 |
| Gradient Boosting | 0.679 | 0.422 | 0.607 ± 0.041 |

**Predictores principales al momento de la inscripción:** distancia al centro, edad del paciente, tipo de apoyo social.
**Impacto Costo-Sensible:** Optimizando el umbral de clasificación (FN=$5.000 vs FP=$50), se estiman ahorros de **~$475.000 por cada 1.000 pacientes** respecto a no intervenir.

---

## 🛠️ Stack Técnico

| Categoría | Herramientas |
|---|---|
| **Lenguaje** | Python 3.9+ |
| **Base de datos** | PostgreSQL (schema + vistas analíticas) |
| **Datos** | Pandas, NumPy, SciPy |
| **Visualización** | Matplotlib, Seaborn, Plotly Express |
| **ML** | Scikit-learn (LR, RandomForest, HistGradientBoosting) |
| **Análisis de supervivencia** | Kaplan-Meier (implementación manual) |
| **Explicabilidad** | LIME (lime-tabular) |
| **Despliegue** | Streamlit |
| **BI Reportes** | Power BI Desktop |

---

## 📊 Dashboard Power BI

El dashboard Power BI conecta directamente con la capa de datos procesados. Tres páginas de reporte:

**Página 1 — Resumen del Programa:** KPIs de abandono, tendencia temporal, distribución de outcomes, segmentadores por región/VIH/año.

**Página 2 — Segmentación de Riesgo:** Mapa por región, histograma de score de riesgo, tabla de pacientes de alto riesgo (top 50).

**Página 3 — Planificador de Intervención:** Top 10 pacientes accionables, calculadora de ahorro de costos (DAX), resumen de drivers LIME, ficha de paciente individual.

**Conexión:** Power BI Desktop → Obtener datos → Texto/CSV → `data/processed/ml_dataset.csv`

---

## 🔬 Clasificación OMS de Resultados de Tratamiento TB

- **Curado:** Confirmado bacteriológicamente al final del tratamiento
- **Tratamiento Completado:** Completado sin confirmación bacteriológica
- **Abandonado (Defaulted):** Tratamiento interrumpido ≥2 meses consecutivos ← *nuestro target*
- **Fracaso del Tratamiento:** Frotis/cultivo positivo en el mes 5+
- **Fallecido:** Cualquier causa durante el tratamiento

---

## 📚 Referencias

1. OMS Reporte Global de Tuberculosis 2025.
2. Zhang, F. "Using Machine Learning Methods to Predict Early Treatment Outcomes for Multidrug-Resistant or Rifampicin-Resistant Tuberculosis to Enhance Patient Cure Rates: Development and Validation of Multiple Models".
3. Wang L. "A multi-stage machine learning framework for stepwise prediction of tuberculosis treatment outcomes: Integrating gradient boosted decision trees and feature-level analysis for clinical decision support".
4. Chen, J."Predictive machine learning models for anticipating loss to follow-up in tuberculosis patients throughout anti-TB treatment journey".

---

## 👤 Autor

**Facundo Colaccini, PhD** — Ciencias Biológicas (TB / Drug Discovery) → Data Science

- 🔗 [LinkedIn](www.linkedin.com/in/facundo-colaccini)
- 💻 [GitHub](https://github.com/facuccini)
- 🔬 [Ver también: Proyecto A — FasR Drug Discovery Pipeline](../fasr-drug-discovery/)



