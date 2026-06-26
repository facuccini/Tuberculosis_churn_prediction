-- ══════════════════════════════════════════════════════════════
-- 02_analytical_queries.sql
-- Analytical queries for the TB dropout prediction pipeline
-- Database: PostgreSQL 14+
--
-- These queries build the "master table" that feeds the ML model.
-- They demonstrate: JOINs, CTEs, CASE, aggregations, window functions.
-- ══════════════════════════════════════════════════════════════


-- ════════════════════════════════════════════════════════
-- QUERY 1: Analytical view — Per-patient summary
-- Builds the feature table for the ML model
-- Joins all 4 tables with JOINs and aggregates visit variables
-- ════════════════════════════════════════════════════════
CREATE OR REPLACE VIEW patient_analytics AS

WITH

-- ─── CTE 1: Adherence metrics per patient ────────────────────
visit_metrics AS (
    SELECT
        patient_id,
        COUNT(*)                                    AS total_visits_scheduled,
        SUM(CASE WHEN attendance = TRUE THEN 1 ELSE 0 END)
                                                    AS visits_attended,
        SUM(CASE WHEN attendance = FALSE THEN 1 ELSE 0 END)
                                                    AS visits_missed,
        ROUND(
            100.0 * SUM(CASE WHEN attendance THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0),
            2
        )                                           AS attendance_rate_pct,
        -- Maximum consecutive missed visits (early warning signal)
        MAX(CASE WHEN attendance = FALSE THEN 1 ELSE 0 END)
                                                    AS had_any_absence,
        AVG(adherence_score)                        AS avg_adherence_score,
        -- Severe adverse effects (common cause of dropout)
        SUM(CASE WHEN side_effects IN ('Moderate', 'Severe') THEN 1 ELSE 0 END)
                                                    AS n_severe_side_effects,
        -- Social worker visit (protective intervention)
        SUM(CASE WHEN social_worker_visit THEN 1 ELSE 0 END)
                                                    AS n_social_worker_visits,
        -- Weight trajectory (proxy for treatment response)
        MIN(weight_kg)                              AS min_weight,
        MAX(weight_kg)                              AS max_weight,
        (MAX(weight_kg) - MIN(weight_kg))           AS weight_gain_kg
    FROM visits
    GROUP BY patient_id
),

-- ─── CTE 2: Bacteriological result at month 2 ────────────────
-- Positive smear at month 2 = predictor of treatment failure
lab_month2 AS (
    SELECT
        patient_id,
        smear_result    AS smear_month2,
        culture_result  AS culture_month2
    FROM results
    WHERE test_month = 2
),

-- ─── CTE 3: Bacteriological result at month 5 ────────────────
lab_month5 AS (
    SELECT
        patient_id,
        smear_result    AS smear_month5,
        culture_result  AS culture_month5
    FROM results
    WHERE test_month = 5
),

-- ─── CTE 4: Enrollment baseline result ───────────────────────
lab_baseline AS (
    SELECT
        patient_id,
        smear_result    AS smear_baseline,
        xray_result     AS xray_baseline,
        cd4_count       AS cd4_baseline,
        viral_load      AS viral_load_baseline
    FROM results
    WHERE test_month = 0
)

-- ─── FINAL QUERY: JOIN all CTEs ──────────────────────────────
SELECT
    -- IDs and dates
    p.patient_id,
    p.registration_date,
    t.start_date                                    AS treatment_start,
    t.end_date                                      AS treatment_end,

    -- Demographics
    p.age,
    p.sex,
    p.region,
    p.education_level,
    p.employment_status,
    p.distance_km,
    p.household_contacts,

    -- Clinical risk factors
    p.hiv_status,
    p.diabetes,
    p.alcohol_use,
    p.smoking,
    p.previous_tb,
    p.tb_type,
    p.drug_resistance,

    -- Treatment
    t.regimen,
    t.supporter,
    t.dot_method,
    t.outcome,
    t.dropout_month,

    -- TARGET VARIABLE: Dropout (1 = defaulted, 0 = completed/cured)
    CASE WHEN t.outcome = 'Defaulted' THEN 1 ELSE 0 END
                                                    AS dropout_label,

    -- Visit metrics (behavioral features)
    vm.total_visits_scheduled,
    vm.visits_attended,
    vm.visits_missed,
    vm.attendance_rate_pct,
    vm.avg_adherence_score,
    vm.n_severe_side_effects,
    vm.n_social_worker_visits,
    vm.weight_gain_kg,

    -- Laboratory results
    lb.smear_baseline,
    lb.xray_baseline,
    lb.cd4_baseline,
    lb.viral_load_baseline,
    l2.smear_month2,
    l5.smear_month5,

    -- Derived features
    EXTRACT(YEAR FROM AGE(p.registration_date, ('1900-01-01'::date + p.age * INTERVAL '1 year')))
                                                    AS birth_year_approx,
    CASE WHEN p.distance_km > 20 THEN 1 ELSE 0 END AS far_from_clinic,
    CASE WHEN p.hiv_status = 'Positive' THEN 1 ELSE 0 END
                                                    AS hiv_positive,
    CASE WHEN vm.attendance_rate_pct < 70 THEN 1 ELSE 0 END
                                                    AS low_attendance_flag,
    DATE_PART('days', t.end_date - t.start_date)    AS treatment_duration_days

FROM patients p
LEFT JOIN treatments t      ON p.patient_id = t.patient_id
LEFT JOIN visit_metrics vm  ON p.patient_id = vm.patient_id
LEFT JOIN lab_baseline lb   ON p.patient_id = lb.patient_id
LEFT JOIN lab_month2 l2     ON p.patient_id = l2.patient_id
LEFT JOIN lab_month5 l5     ON p.patient_id = l5.patient_id

-- Only patients with completed or defaulted treatment (exclude "Still on Treatment")
WHERE t.outcome NOT IN ('Still on Treatment', 'Transferred Out')
  AND t.outcome IS NOT NULL;

COMMENT ON VIEW patient_analytics IS
'Master analytical view: joins patients, visits, results, and treatments.
 Feature table ready for the dropout prediction (churn) model.';


-- ════════════════════════════════════════════════════════
-- QUERY 2: Operational KPIs for Dashboard (Power BI)
-- ════════════════════════════════════════════════════════
CREATE OR REPLACE VIEW dashboard_kpis AS
SELECT
    -- KPI 1: Global dropout rate
    ROUND(100.0 * SUM(CASE WHEN outcome = 'Defaulted' THEN 1 ELSE 0 END)
          / NULLIF(COUNT(*), 0), 2)                 AS dropout_rate_pct,

    -- KPI 2: Therapeutic success rate (cured + treatment completed)
    ROUND(100.0 * SUM(CASE WHEN outcome IN ('Cured', 'Treatment Completed') THEN 1 ELSE 0 END)
          / NULLIF(COUNT(*), 0), 2)                 AS success_rate_pct,

    -- KPI 3: Total patient count
    COUNT(*)                                        AS total_patients,

    -- KPI 4: Deaths
    SUM(CASE WHEN outcome = 'Died' THEN 1 ELSE 0 END)
                                                    AS n_deaths,

    -- KPI 5: Mean month of dropout
    ROUND(AVG(dropout_month), 1)                    AS avg_dropout_month,

    -- KPI 6: % HIV co-infected
    ROUND(100.0 * SUM(CASE WHEN p.hiv_status = 'Positive' THEN 1 ELSE 0 END)
          / NULLIF(COUNT(*), 0), 2)                 AS hiv_coinfection_pct

FROM treatments t
JOIN patients p ON t.patient_id = p.patient_id
WHERE t.outcome NOT IN ('Still on Treatment', 'Transferred Out');


-- ════════════════════════════════════════════════════════
-- QUERY 3: Dropout by region and age (for dashboard filters)
-- ════════════════════════════════════════════════════════
CREATE OR REPLACE VIEW dropout_by_region AS
SELECT
    p.region,
    COUNT(*)                                                    AS total_patients,
    SUM(CASE WHEN t.outcome = 'Defaulted' THEN 1 ELSE 0 END)   AS n_dropouts,
    ROUND(
        100.0 * SUM(CASE WHEN t.outcome = 'Defaulted' THEN 1 ELSE 0 END)
              / NULLIF(COUNT(*), 0),
        2
    )                                                           AS dropout_rate_pct,
    ROUND(AVG(p.age), 1)                                        AS avg_age,
    ROUND(AVG(p.distance_km), 1)                                AS avg_distance_km,
    SUM(CASE WHEN p.hiv_status = 'Positive' THEN 1 ELSE 0 END) AS n_hiv_positive
FROM patients p
JOIN treatments t ON p.patient_id = t.patient_id
WHERE t.outcome NOT IN ('Still on Treatment', 'Transferred Out')
GROUP BY p.region
ORDER BY dropout_rate_pct DESC;


-- ════════════════════════════════════════════════════════
-- QUERY 4: Temporal analysis — Month with highest dropout
-- ════════════════════════════════════════════════════════
SELECT
    dropout_month,
    COUNT(*)                                        AS n_dropouts,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2)
                                                    AS pct_of_total_dropouts,
    -- Cumulative
    SUM(COUNT(*)) OVER (ORDER BY dropout_month)     AS cumulative_dropouts
FROM treatments
WHERE outcome = 'Defaulted'
  AND dropout_month IS NOT NULL
GROUP BY dropout_month
ORDER BY dropout_month;

-- ════════════════════════════════════════════════════════
-- QUERY 5: Top 10 highest-risk patients
-- For health workers (preventive action)
-- Updated with ML model predictions after each inference run
-- ════════════════════════════════════════════════════════
-- Note: "predicted_dropout_prob" column is inserted by the Python inference pipeline.

CREATE TABLE IF NOT EXISTS ml_predictions (
    prediction_id       SERIAL PRIMARY KEY,
    patient_id          INTEGER REFERENCES patients(patient_id),
    prediction_date     DATE NOT NULL DEFAULT CURRENT_DATE,
    dropout_probability NUMERIC(6,4) NOT NULL,
    risk_tier           VARCHAR(10) CHECK (risk_tier IN ('HIGH', 'MEDIUM', 'LOW')),
    top_risk_factor     VARCHAR(100),   -- LIME output (local explanation)
    created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Dashboard view: Top 10 highest-risk patients
CREATE OR REPLACE VIEW high_risk_patients AS
SELECT
    mp.patient_id,
    p.age,
    p.sex,
    p.region,
    p.hiv_status,
    p.distance_km,
    mp.dropout_probability,
    mp.risk_tier,
    mp.top_risk_factor,
    mp.prediction_date
FROM ml_predictions mp
JOIN patients p ON mp.patient_id = p.patient_id
WHERE mp.risk_tier = 'HIGH'
  AND mp.prediction_date = CURRENT_DATE
ORDER BY mp.dropout_probability DESC
LIMIT 10;

COMMENT ON VIEW high_risk_patients IS
'Top 10 patients at highest dropout risk TODAY.
 Update daily by running: python src/inference_pipeline.py';
