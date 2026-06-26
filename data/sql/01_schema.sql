-- ══════════════════════════════════════════════════════════════
-- schema.sql
-- Relational schema for the TB patient tracking system
-- Database: PostgreSQL 14+
--
-- Tables:
--   patients   → Demographics and clinical baseline
--   visits     → Monthly clinic visit records
--   results    → Laboratory results (smear, culture, X-ray)
--   treatments → Assigned treatment regimens
-- ══════════════════════════════════════════════════════════════

-- ─────────────────────────────────────────────────────────────
-- TABLE 1: PATIENTS
-- Demographics and baseline clinical characteristics at enrollment
-- ─────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS patients (
    patient_id          SERIAL PRIMARY KEY,
    registration_date   DATE NOT NULL,
    age                 INTEGER NOT NULL CHECK (age BETWEEN 0 AND 120),
    sex                 VARCHAR(10) NOT NULL CHECK (sex IN ('Male', 'Female')),
    region              VARCHAR(50) NOT NULL,
    district            VARCHAR(100),
    education_level     VARCHAR(30) CHECK (education_level IN
                            ('None', 'Primary', 'Secondary', 'Tertiary')),
    employment_status   VARCHAR(30) CHECK (employment_status IN
                            ('Employed', 'Unemployed', 'Informal', 'Student', 'Retired')),
    hiv_status          VARCHAR(20) CHECK (hiv_status IN
                            ('Positive', 'Negative', 'Unknown')),
    diabetes            BOOLEAN DEFAULT FALSE,
    alcohol_use         VARCHAR(20) CHECK (alcohol_use IN ('None', 'Occasional', 'Heavy')),
    smoking             VARCHAR(20) CHECK (smoking IN ('Never', 'Former', 'Current')),
    distance_km         NUMERIC(6,2),                     -- distance to the health clinic
    household_contacts  INTEGER DEFAULT 0,                -- number of household members
    previous_tb         BOOLEAN DEFAULT FALSE,            -- prior TB episode (relapse)
    tb_type             VARCHAR(30) CHECK (tb_type IN
                            ('Pulmonary', 'Extrapulmonary', 'Both')),
    drug_resistance     VARCHAR(20) CHECK (drug_resistance IN
                            ('Sensitive', 'MDR', 'XDR', 'Unknown')),
    created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

COMMENT ON TABLE patients IS 'Baseline demographic and clinical data for TB patients at treatment start';
COMMENT ON COLUMN patients.distance_km IS 'Distance in km from patient residence to health clinic';
COMMENT ON COLUMN patients.hiv_status IS 'HIV is the primary co-factor for TB treatment dropout';

-- ─────────────────────────────────────────────────────────────
-- TABLE 2: VISITS
-- Monthly attendance records for scheduled control visits
-- ─────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS visits (
    visit_id            SERIAL PRIMARY KEY,
    patient_id          INTEGER NOT NULL REFERENCES patients(patient_id) ON DELETE CASCADE,
    visit_date          DATE NOT NULL,
    visit_month         INTEGER NOT NULL CHECK (visit_month BETWEEN 1 AND 24),
    attendance          BOOLEAN NOT NULL,               -- TRUE = attended, FALSE = missed
    reason_absence      VARCHAR(100),                   -- only if attendance = FALSE
    weight_kg           NUMERIC(5,2),
    adherence_score     INTEGER CHECK (adherence_score BETWEEN 0 AND 100),
                                                        -- % of doses taken (self-reported)
    side_effects        VARCHAR(50) CHECK (side_effects IN
                            ('None', 'Mild', 'Moderate', 'Severe')),
    social_worker_visit BOOLEAN DEFAULT FALSE,
    created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_visits_patient_id ON visits(patient_id);
CREATE INDEX idx_visits_date ON visits(visit_date);

COMMENT ON TABLE visits IS 'Monthly scheduled visit records during the treatment period';
COMMENT ON COLUMN visits.adherence_score IS 'Adherence score 0-100 based on pill count and self-report';

-- ─────────────────────────────────────────────────────────────
-- TABLE 3: LABORATORY RESULTS
-- Smear, culture, and radiology at key treatment checkpoints
-- ─────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS results (
    result_id           SERIAL PRIMARY KEY,
    patient_id          INTEGER NOT NULL REFERENCES patients(patient_id) ON DELETE CASCADE,
    result_date         DATE NOT NULL,
    test_month          INTEGER CHECK (test_month IN (0, 2, 5, 6, 8)),
                                                        -- standard control months
    smear_result        VARCHAR(20) CHECK (smear_result IN
                            ('Negative', '+', '++', '+++')),
    culture_result      VARCHAR(20) CHECK (culture_result IN
                            ('Negative', 'Positive', 'Contaminated', 'Pending')),
    xray_result         VARCHAR(30) CHECK (xray_result IN
                            ('Improved', 'Stable', 'Worsened', 'Not done')),
    cd4_count           INTEGER,                        -- HIV-positive patients only
    viral_load          INTEGER,                        -- copies/mL, HIV-positive patients only
    created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_results_patient_id ON results(patient_id);

COMMENT ON TABLE results IS 'Laboratory results at standard treatment control checkpoints';

-- ─────────────────────────────────────────────────────────────
-- TABLE 4: TREATMENTS AND OUTCOMES
-- Assigned regimen and final treatment outcome
-- ─────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS treatments (
    treatment_id        SERIAL PRIMARY KEY,
    patient_id          INTEGER NOT NULL REFERENCES patients(patient_id) ON DELETE CASCADE,
    start_date          DATE NOT NULL,
    end_date            DATE,
    regimen             VARCHAR(20) NOT NULL CHECK (regimen IN
                            ('2HRZE/4HR',          -- standard first-line (6 months)
                             '2HRZE/4HR3',         -- intermittent first-line
                             'HRZE/HRE',           -- 8-month variant
                             'BPaL',               -- MDR: bedaquiline-pretomanid-linezolid
                             'BDQ/DLM/LZD',        -- XDR
                             'Other')),
    treatment_center    VARCHAR(100),
    supporter           VARCHAR(30) CHECK (supporter IN
                            ('Family', 'CHW', 'Clinic', 'None')),
                                                        -- who supports DOT (Directly Observed Therapy)
    dot_method          VARCHAR(30) CHECK (dot_method IN
                            ('In-person', 'Video DOT', 'Self-administered')),
    outcome             VARCHAR(30) CHECK (outcome IN
                            ('Cured',              -- bacteriologically confirmed
                             'Treatment Completed',-- without bacteriological confirmation
                             'Defaulted',          -- DROPOUT (>2 months without medication)
                             'Treatment Failed',   -- positive at month 5+
                             'Died',
                             'Transferred Out',
                             'Still on Treatment')),
    dropout_month       INTEGER,                        -- treatment month when dropout occurred
    created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

COMMENT ON TABLE treatments IS 'Treatment regimen and final outcome per WHO classification';
COMMENT ON COLUMN treatments.outcome IS 'WHO classification: "Defaulted" = dropout (>2 months without taking medication)';
COMMENT ON COLUMN treatments.dropout_month IS 'Treatment month when dropout occurred (NULL if patient did not default)';
