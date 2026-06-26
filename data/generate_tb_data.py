# -*- coding: utf-8 -*-
"""
generate_tb_data.py
-------------------
Generates a realistic synthetic dataset of TB patients with demographic,
clinical, and treatment adherence data. Simulates the 4 relational tables
defined in the SQL schema.

Based on real-world TB program statistics (WHO, PAHO):
- Global default rate: ~15-20%
- Risk factors: HIV, distance, young age, alcohol, prior TB
- Temporal distribution: peak dropout in treatment months 2-4

Usage: python data/generate_tb_data.py
"""

import pandas as pd
import numpy as np
import random
from datetime import date, timedelta

random.seed(42)
np.random.seed(42)

# ─────────────────────────────────────────────────────────────
# SIMULATION PARAMETERS
# ─────────────────────────────────────────────────────────────
N_PATIENTS = 800  # 3-year cohort (2021–2023)

REGIONS = {
    'North':   {'base_dropout': 0.22, 'avg_distance': 35, 'hiv_prev': 0.18},
    'South':   {'base_dropout': 0.12, 'avg_distance': 12, 'hiv_prev': 0.08},
    'East':    {'base_dropout': 0.18, 'avg_distance': 25, 'hiv_prev': 0.14},
    'West':    {'base_dropout': 0.20, 'avg_distance': 30, 'hiv_prev': 0.16},
    'Central': {'base_dropout': 0.10, 'avg_distance': 8,  'hiv_prev': 0.10},
}

REGIMENS = ['2HRZE/4HR', '2HRZE/4HR', '2HRZE/4HR',  # 75% first-line
            '2HRZE/4HR3', 'HRZE/HRE', 'BPaL']


def calc_dropout_probability(patient: dict) -> float:
    """
    Calculates dropout probability based on known risk factors.
    Based on meta-analyses (Tola et al., 2019; WHO Global TB Report 2023).
    """
    region = REGIONS[patient['region']]
    p = region['base_dropout']

    # Risk-INCREASING factors (known ORs from literature)
    if patient['hiv_status'] == 'Positive':    p += 0.08   # OR ~1.5
    if patient['distance_km'] > 20:            p += 0.06   # OR ~1.4
    if patient['age'] < 30:                    p += 0.05   # Young patients: lower adherence
    if patient['alcohol_use'] == 'Heavy':      p += 0.10   # OR ~2.0
    if patient['employment_status'] == 'Unemployed': p += 0.05
    if patient['previous_tb']:                 p += 0.04   # Relapse = higher risk
    if patient['drug_resistance'] in ('MDR', 'XDR'): p += 0.08  # Longer/more toxic regimen
    if patient['education_level'] == 'No Formal': p += 0.04
    if patient['smoking'] == 'Current':        p += 0.03

    # PROTECTIVE factors
    if patient['supporter'] == 'CHW':          p -= 0.07   # Community Health Worker
    if patient['supporter'] == 'Family':       p -= 0.04
    if patient['dot_method'] == 'In-person':   p -= 0.05
    if patient['education_level'] == 'Tertiary': p -= 0.04

    return max(0.02, min(0.95, p))  # Clamp between 2% and 95%


def dropout_month_distribution(base_months: int = 6) -> int:
    """
    Temporal distribution of dropout: concentrated in months 2-4.
    The intensive treatment phase has more side effects and frequent visits.
    """
    weights = {1: 0.08, 2: 0.20, 3: 0.22, 4: 0.18,
               5: 0.12, 6: 0.10, 7: 0.05, 8: 0.05}
    months = list(weights.keys())
    probs = list(weights.values())
    return random.choices(months, weights=probs)[0]


def generate_patients(n: int) -> pd.DataFrame:
    """Generates the patients table."""
    records = []
    start_date = date(2021, 1, 1)
    end_date = date(2023, 12, 31)

    for i in range(n):
        region_name = random.choices(
            list(REGIONS.keys()),
            weights=[0.20, 0.20, 0.20, 0.20, 0.20]
        )[0]
        region = REGIONS[region_name]

        reg_date = start_date + timedelta(
            days=random.randint(0, (end_date - start_date).days)
        )

        hiv = 'Positive' if random.random() < region['hiv_prev'] else \
              ('Unknown' if random.random() < 0.05 else 'Negative')

        age = int(np.random.choice(
            range(15, 80),
            p=np.exp(-0.015 * np.abs(np.arange(15, 80) - 38)) /
              np.exp(-0.015 * np.abs(np.arange(15, 80) - 38)).sum()
        ))

        records.append({
            'patient_id': i + 1,
            'registration_date': reg_date,
            'age': age,
            'sex': random.choices(['Male', 'Female'], weights=[0.58, 0.42])[0],
            'region': region_name,
            'district': f"District_{random.randint(1, 8):02d}",
            'education_level': random.choices(
                ['No Formal', 'Primary', 'Secondary', 'Tertiary'],
                weights=[0.10, 0.35, 0.40, 0.15]
            )[0],
            'employment_status': random.choices(
                ['Employed', 'Unemployed', 'Informal', 'Student', 'Retired'],
                weights=[0.30, 0.25, 0.28, 0.10, 0.07]
            )[0],
            'hiv_status': hiv,
            'diabetes': random.random() < 0.08,
            'alcohol_use': random.choices(
                ['Non-drinker', 'Occasional', 'Heavy'],
                weights=[0.55, 0.30, 0.15]
            )[0],
            'smoking': random.choices(
                ['Never', 'Former', 'Current'],
                weights=[0.50, 0.25, 0.25]
            )[0],
            'distance_km': round(max(0.5, np.random.lognormal(
                mean=np.log(region['avg_distance']), sigma=0.7
            )), 1),
            'household_contacts': max(0, int(np.random.poisson(3))),
            'previous_tb': random.random() < 0.12,
            'tb_type': random.choices(
                ['Pulmonary', 'Extrapulmonary', 'Both'],
                weights=[0.75, 0.20, 0.05]
            )[0],
            'drug_resistance': random.choices(
                ['Sensitive', 'MDR', 'XDR', 'Unknown'],
                weights=[0.82, 0.12, 0.02, 0.04]
            )[0],
        })

    df = pd.DataFrame(records)

    # -- Introduce realistic missing values (MCAR / MAR patterns) ----------
    # Real TB program records always have some incompleteness at intake.
    # Rates calibrated to field data from LMIC settings (Dangisso et al. 2015).
    rng = np.random.default_rng(42)

    # distance_km: ~5% - rural patients estimate distance, not always recorded
    df.loc[rng.random(len(df)) < 0.05, 'distance_km'] = np.nan

    # household_contacts: ~4% - not always captured during registration
    df.loc[rng.random(len(df)) < 0.04, 'household_contacts'] = np.nan

    # alcohol_use: ~8% - self-reported, stigmatised; frequently left blank
    df.loc[rng.random(len(df)) < 0.08, 'alcohol_use'] = np.nan

    # education_level: ~5% - especially in older / rural patients
    df.loc[rng.random(len(df)) < 0.05, 'education_level'] = np.nan

    return df


def generate_treatments(df_patients: pd.DataFrame) -> pd.DataFrame:
    """Generates the treatments table with outcomes."""
    records = []

    for _, p in df_patients.iterrows():
        # Assign regimen based on drug resistance
        if p['drug_resistance'] == 'MDR':
            regimen = 'BPaL'
            tto_months = 18
        elif p['drug_resistance'] == 'XDR':
            regimen = 'BDQ/DLM/LZD'
            tto_months = 24
        else:
            regimen = random.choices(['2HRZE/4HR', '2HRZE/4HR3', 'HRZE/HRE'],
                                     weights=[0.80, 0.12, 0.08])[0]
            tto_months = 6 if '4HR' in regimen else 8

        supporter = random.choices(
            ['Family', 'CHW', 'Clinic', 'None'],
            weights=[0.40, 0.20, 0.30, 0.10]
        )[0]
        dot_method = random.choices(
            ['In-person', 'Video DOT', 'Self-administered'],
            weights=[0.55, 0.15, 0.30]
        )[0]

        # Build temp dict to calculate dropout probability
        p_dict = p.to_dict()
        p_dict['supporter'] = supporter
        p_dict['dot_method'] = dot_method

        dropout_prob = calc_dropout_probability(p_dict)
        start = p['registration_date']

        # Determine outcome
        rand = random.random()
        if rand < dropout_prob:
            outcome = 'Defaulted'
            d_month = dropout_month_distribution(tto_months)
            end = start + timedelta(days=d_month * 30 + random.randint(-7, 7))
        elif rand < dropout_prob + 0.04:
            outcome = 'Died'
            d_month = None
            end = start + timedelta(days=random.randint(30, tto_months * 30))
        elif rand < dropout_prob + 0.06:
            outcome = 'Treatment Failed'
            d_month = None
            end = start + timedelta(days=tto_months * 30 + 30)
        elif rand < dropout_prob + 0.08:
            outcome = 'Treatment Completed'
            d_month = None
            end = start + timedelta(days=tto_months * 30 + random.randint(-14, 14))
        else:
            outcome = 'Cured'
            d_month = None
            end = start + timedelta(days=tto_months * 30 + random.randint(-14, 14))

        records.append({
            'treatment_id': p['patient_id'],
            'patient_id': p['patient_id'],
            'start_date': start,
            'end_date': end,
            'regimen': regimen,
            'treatment_center': f"Center_{p['region']}_{random.randint(1,3)}",
            'supporter': supporter,
            'dot_method': dot_method,
            'outcome': outcome,
            'dropout_month': d_month,
        })

    return pd.DataFrame(records)


def generate_visits(df_patients: pd.DataFrame, df_treatments: pd.DataFrame) -> pd.DataFrame:
    """Generates monthly visit records."""
    records = []
    visit_id = 1
    side_effects_options = ['None', 'Mild', 'Moderate', 'Severe']

    reasons_absence = [
        'Work conflict', 'Transport unavailable', 'Felt better',
        'Side effects', 'Forgot', 'Family emergency', 'Financial',
        'Distance too far', 'No childcare', 'Unknown'
    ]

    merged = df_treatments.merge(df_patients[['patient_id', 'age', 'hiv_status',
                                               'distance_km', 'alcohol_use']],
                                  on='patient_id')

    for _, row in merged.iterrows():
        is_dropout = row['outcome'] == 'Defaulted'
        dropout_mo = row['dropout_month'] if is_dropout else 99

        # Calculate treatment duration in months
        tto_months = 6 if row['regimen'] in ['2HRZE/4HR', '2HRZE/4HR3'] else 8
        if row['regimen'] in ['BPaL', 'BDQ/DLM/LZD']:
            tto_months = 18

        base_weight = 55 + random.gauss(10, 8)
        weight = base_weight

        for month in range(1, tto_months + 1):
            # No visits after the dropout month
            if is_dropout and month > dropout_mo:
                break

            visit_date = row['start_date'] + timedelta(days=month * 30 + random.randint(-3, 3))

            # ── Attendance probability ────────────────────────────
            # REALISTIC: attendance only drops in the final month(s) before dropout.
            # Patients who will eventually default behave similarly to completers in
            # early months — this is what makes early prediction clinically valuable.
            if is_dropout:
                months_to_dropout = dropout_mo - month
                if months_to_dropout == 0:
                    att_prob = 0.30   # Dropout month: sharply reduced attendance
                elif months_to_dropout == 1:
                    att_prob = 0.62   # Month prior: moderate signal
                else:
                    att_prob = 0.80   # Early months: overlapping with completers
            else:
                att_prob = 0.90 if row['distance_km'] <= 15 else 0.79

            attended = random.random() < att_prob

            # Weight: improves over time for completers, worse for HIV or adverse effects
            if attended:
                weight_change = random.gauss(0.3, 0.5)
                if row['hiv_status'] == 'Positive':
                    weight_change -= 0.2
                weight = max(35, weight + weight_change)

            side_eff = random.choices(
                side_effects_options,
                weights=[0.50, 0.30, 0.15, 0.05]
            )[0]

            # ── Adherence score ───────────────────────────────────
            # REALISTIC: high variability, strong signal only at dropout month and
            # the one prior. Early months: similar between groups with high overlap.
            if not is_dropout:
                adh_base = 87
            else:
                months_to_dropout = dropout_mo - month
                if months_to_dropout == 0:
                    adh_base = 42   # Dropout month: sharp decline
                elif months_to_dropout == 1:
                    adh_base = 63   # Month prior: moderate decline
                else:
                    adh_base = 77   # Early months: slight difference
            if side_eff in ('Moderate', 'Severe'):
                adh_base -= 12

            records.append({
                'visit_id': visit_id,
                'patient_id': row['patient_id'],
                'visit_date': visit_date,
                'visit_month': month,
                'attendance': attended,
                'reason_absence': random.choice(reasons_absence) if not attended else None,
                'weight_kg': round(weight, 1) if attended else None,
                'adherence_score': min(100, max(0, int(random.gauss(adh_base, 12)))),
                'side_effects': side_eff,
                'social_worker_visit': random.random() < 0.12,
            })
            visit_id += 1

    return pd.DataFrame(records)


def generate_results(df_patients: pd.DataFrame, df_treatments: pd.DataFrame) -> pd.DataFrame:
    """Generates laboratory results at pre-defined control checkpoints."""
    records = []
    result_id = 1
    control_months = [0, 2, 5]

    merged = df_treatments.merge(df_patients[['patient_id', 'hiv_status', 'tb_type']],
                                  on='patient_id')

    for _, row in merged.iterrows():
        is_dropout = row['outcome'] == 'Defaulted'
        dropout_mo = row['dropout_month'] if is_dropout else 99

        for test_month in control_months:
            # No results after the dropout month
            if is_dropout and test_month >= dropout_mo:
                continue

            # Sputum smear: positive at baseline, improves with treatment
            if test_month == 0:
                smear = random.choices(['+', '++', '+++'], weights=[0.35, 0.40, 0.25])[0]
            elif test_month == 2:
                # ~80% smear conversion by month 2 in responders
                if row['outcome'] in ('Cured', 'Treatment Completed'):
                    smear = random.choices(['Negative', '+', '++'],
                                           weights=[0.80, 0.15, 0.05])[0]
                else:
                    smear = random.choices(['Negative', '+', '++'],
                                           weights=[0.50, 0.30, 0.20])[0]
            else:  # month 5
                smear = 'Negative' if random.random() < 0.92 else '+'

            # HIV-positive patients: add CD4 count and viral load
            cd4 = None
            viral_load = None
            if row['hiv_status'] == 'Positive':
                cd4 = max(50, int(np.random.lognormal(np.log(250), 0.5)))
                viral_load = int(np.random.lognormal(np.log(10000), 1.0))

            records.append({
                'result_id': result_id,
                'patient_id': row['patient_id'],
                'result_date': row['start_date'] + timedelta(days=test_month * 30),
                'test_month': test_month,
                'smear_result': smear,
                'culture_result': random.choices(
                    ['Negative', 'Positive', 'Contaminated'],
                    weights=[0.70, 0.25, 0.05]
                )[0] if test_month >= 2 else 'Positive',
                'xray_result': random.choices(
                    ['Improved', 'Stable', 'Worsened', 'Not done'],
                    weights=[0.55, 0.25, 0.10, 0.10]
                )[0],
                'cd4_count': cd4,
                'viral_load': viral_load,
            })
            result_id += 1

    return pd.DataFrame(records)


def main():
    print("=" * 55)
    print("  Generating synthetic TB patient dataset")
    print("=" * 55)

    # 1. Patients
    print(f"\n[1/4] Generating {N_PATIENTS} patients...")
    df_patients = generate_patients(N_PATIENTS)

    # 2. Treatments + Outcomes
    print("[2/4] Generating treatments and outcomes...")
    df_treatments = generate_treatments(df_patients)

    # 3. Visits
    print("[3/4] Generating monthly visit records...")
    df_visits = generate_visits(df_patients, df_treatments)

    # 4. Lab results
    print("[4/4] Generating laboratory results...")
    df_results = generate_results(df_patients, df_treatments)

    # ── Save CSVs ─────────────────────────────────────────────
    import os
    os.makedirs('raw', exist_ok=True)

    df_patients.to_csv('raw/patients.csv', index=False)
    df_treatments.to_csv('raw/treatments.csv', index=False)
    df_visits.to_csv('raw/visits.csv', index=False)
    df_results.to_csv('raw/results.csv', index=False)

    print(f'  Patients    : {len(df_patients):,}')
    print(f'  Treatments  : {len(df_treatments):,}')
    print(f'  Visits      : {len(df_visits):,}')
    print(f'  Lab results : {len(df_results):,}')
    dropout_n = (df_treatments['outcome'] == 'Defaulted').sum()
    print(f'  Dropout rate: {dropout_n}/{len(df_treatments)} = {dropout_n/len(df_treatments)*100:.1f}%')
    missing = df_patients.isnull().sum()
    missing = missing[missing > 0]
    if len(missing):
        print('  Missing values:')
        for col, n in missing.items():
            print(f'    {col}: {n} ({n/len(df_patients)*100:.1f}%)')
    print('\nDone. Files written to raw/')


if __name__ == '__main__':
    main()
