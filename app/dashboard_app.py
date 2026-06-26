"""
dashboard_app.py
----------------
Analytical dashboard for TB Treatment Dropout Prediction.
Python equivalent of a Power BI dashboard, implemented with Streamlit.

Structure:
  Tab 1 → Operational KPIs (dropout rate, success rate, deaths)
  Tab 2 → Regional Analysis (filters: region, age, HIV)
  Tab 3 → Top 10 High-Risk Patients (model predictions)
  Tab 4 → Individual patient risk prediction

Deployment: streamlit run app/dashboard_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import sys
import os
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# ════════════════════════════════════════════════════════════
# PAGE CONFIG
# ════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="TB Churn Dashboard | Treatment Dropout Prediction",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ════════════════════════════════════════════════════════════
# THEME CSS — Corporate Power BI-style
# ════════════════════════════════════════════════════════════
st.markdown("""
<style>
    .main { background-color: #F4F6F9; }
    .kpi-card {
        background: white;
        border-radius: 12px;
        padding: 1.2rem 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border-left: 5px solid #1E3A5F;
        text-align: center;
    }
    .kpi-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A5F;
        line-height: 1.1;
    }
    .kpi-label {
        font-size: 0.82rem;
        color: #7F8C8D;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .kpi-delta-pos { color: #27AE60; font-size: 0.9rem; }
    .kpi-delta-neg { color: #E74C3C; font-size: 0.9rem; }
    .risk-high { color: #E74C3C; font-weight: bold; }
    .risk-medium { color: #F39C12; font-weight: bold; }
    .risk-low { color: #27AE60; font-weight: bold; }
    .section-header {
        font-size: 1.1rem; font-weight: 600;
        color: #2C3E50; border-bottom: 2px solid #E8ECEF;
        padding-bottom: 0.4rem; margin-bottom: 1rem;
    }
    div[data-testid="metric-container"] {
        background: white;
        border-radius: 10px;
        padding: 0.8rem;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    }
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# DATA LOADING (cached)
# ════════════════════════════════════════════════════════════
@st.cache_data
def load_data():
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
    raw_dir  = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')
    rep_dir  = os.path.join(os.path.dirname(__file__), '..', 'reports')

    df = pd.read_csv(os.path.join(data_dir, 'ml_dataset.csv')) \
         if os.path.exists(os.path.join(data_dir, 'ml_dataset.csv')) else None

    top10 = pd.read_csv(os.path.join(rep_dir, 'top10_high_risk_patients.csv')) \
            if os.path.exists(os.path.join(rep_dir, 'top10_high_risk_patients.csv')) else None

    return df, top10


@st.cache_resource
def load_model():
    model_path   = os.path.join(os.path.dirname(__file__), '..', 'models', 'rf_tb_model.pkl')
    feat_path    = os.path.join(os.path.dirname(__file__), '..', 'models', 'feature_names.pkl')
    thresh_path  = os.path.join(os.path.dirname(__file__), '..', 'models', 'optimal_threshold.pkl')
    enc_path     = os.path.join(os.path.dirname(__file__), '..', 'models', 'encoders.pkl')

    if not os.path.exists(model_path):
        return None, None, 0.5, None

    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(feat_path, 'rb') as f:
        features = pickle.load(f)
    with open(thresh_path, 'rb') as f:
        thresh = pickle.load(f)['threshold']
    with open(enc_path, 'rb') as f:
        encoders = pickle.load(f)

    return model, features, thresh, encoders


df, top10 = load_data()
model, feature_names, THRESHOLD, encoders = load_model()


# ════════════════════════════════════════════════════════════
# SIDEBAR — GLOBAL FILTERS
# ════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🫁 TB Churn Dashboard")
    st.markdown("*Treatment Dropout Prediction*")
    st.markdown("---")

    if df is not None:
        st.markdown("### 🔽 Filters")

        selected_regions = st.multiselect(
            "Region:",
            options=sorted(df['region'].dropna().unique()),
            default=sorted(df['region'].dropna().unique()),
        )

        age_range = st.slider("Age range:", 15, 80, (15, 80))

        hiv_filter = st.multiselect(
            "HIV status:",
            options=['Negative', 'Positive', 'Unknown'],
            default=['Negative', 'Positive', 'Unknown'],
        )

        st.markdown("---")
        st.markdown("### ⚙️ Model")
        threshold_adj = st.slider(
            "Risk threshold:",
            0.1, 0.9, float(THRESHOLD), 0.05,
            help="Minimum probability to classify as high risk"
        )
    else:
        selected_regions = []
        age_range = (15, 80)
        hiv_filter = ['Negative', 'Positive', 'Unknown']
        threshold_adj = 0.5

    st.markdown("---")
    st.info("""
    **Model:** Random Forest
    **AUC-ROC:** See Notebook 03
    **Features:** 40 clinical + demographic variables
    **Optimal threshold:** minimizes program cost
    """)

# ════════════════════════════════════════════════════════════
# APPLY FILTERS
# ════════════════════════════════════════════════════════════
if df is not None:
    df_filtered = df[
        (df['region'].isin(selected_regions)) &
        (df['age'] >= age_range[0]) & (df['age'] <= age_range[1]) &
        (df['hiv_status'].isin(hiv_filter))
    ].copy()
else:
    df_filtered = pd.DataFrame()


# ════════════════════════════════════════════════════════════
# HEADER
# ════════════════════════════════════════════════════════════
col_title, col_date = st.columns([4, 1])
with col_title:
    st.markdown("# 🫁 TB Treatment Adherence — Analytics Dashboard")
    st.markdown("*National Tuberculosis Program · Treatment Retention Analysis*")
with col_date:
    from datetime import date
    st.markdown(f"<br><br><small>📅 Updated: {date.today().strftime('%Y-%m-%d')}</small>",
                unsafe_allow_html=True)

st.markdown("---")

# ════════════════════════════════════════════════════════════
# TABS
# ════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Operational KPIs",
    "🗺️ Regional Analysis",
    "🎯 High-Risk Patients",
    "🔍 Individual Prediction"
])


# ════════════════════════════════════════════════════════════
# TAB 1: OPERATIONAL KPIs
# ════════════════════════════════════════════════════════════
with tab1:
    if df_filtered.empty:
        st.warning("No data for the selected filters.")
    else:
        st.markdown('<div class="section-header">📌 Key Program Indicators</div>',
                    unsafe_allow_html=True)

        n_total = len(df_filtered)
        n_dropout = df_filtered['dropout_label'].sum()
        n_success = (df_filtered['outcome'].isin(['Cured', 'Treatment Completed'])).sum()
        n_died    = (df_filtered['outcome'] == 'Died').sum()
        n_failed  = (df_filtered['outcome'] == 'Treatment Failed').sum()
        dropout_rate = n_dropout / n_total * 100
        success_rate = n_success / n_total * 100

        # KPIs
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("👥 Total Patients", f"{n_total:,}")
        c2.metric("❌ Dropout Rate", f"{dropout_rate:.1f}%",
                  delta=f"{dropout_rate - 15:.1f}% vs. WHO target",
                  delta_color="inverse")
        c3.metric("✅ Success Rate", f"{success_rate:.1f}%",
                  delta=f"{success_rate - 85:.1f}% vs. WHO target")
        c4.metric("💀 Deaths", f"{n_died:,}")
        c5.metric("⚠️ Treatment Failures", f"{n_failed:,}")

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-header">📈 Outcome Distribution by Region and Year</div>',
                    unsafe_allow_html=True)

        col_a, col_b = st.columns(2)

        with col_a:
            # Outcome breakdown by region
            outcome_region = df_filtered.groupby(['region', 'outcome']).size().reset_index(name='n')
            fig_outcomes = px.bar(
                outcome_region, x='region', y='n', color='outcome',
                color_discrete_map={
                    'Cured': '#27AE60', 'Treatment Completed': '#2ECC71',
                    'Defaulted': '#E74C3C', 'Died': '#8E44AD',
                    'Treatment Failed': '#F39C12',
                },
                barmode='stack',
                title='Outcomes by Region',
                labels={'n': 'N Patients', 'region': 'Region'},
            )
            fig_outcomes.update_layout(height=350, legend_title='Outcome')
            st.plotly_chart(fig_outcomes, use_container_width=True)

        with col_b:
            # Annual dropout trend
            if 'start_date' in df_filtered.columns:
                df_filtered['start_year'] = pd.to_datetime(df_filtered['start_date']).dt.year
                trend = df_filtered.groupby('start_year').agg(
                    total=('dropout_label', 'count'),
                    dropouts=('dropout_label', 'sum')
                ).reset_index()
                trend['dropout_rate'] = trend['dropouts'] / trend['total'] * 100

                fig_trend = go.Figure()
                fig_trend.add_trace(go.Bar(x=trend['start_year'], y=trend['total'],
                                            name='Total patients', marker_color='#BDC3C7'))
                fig_trend.add_trace(go.Scatter(x=trend['start_year'], y=trend['dropout_rate'],
                                                name='Dropout rate (%)', yaxis='y2',
                                                mode='lines+markers', marker_color='#E74C3C',
                                                line=dict(width=2)))
                fig_trend.update_layout(
                    title='Annual Dropout Trend',
                    yaxis=dict(title='N Patients'),
                    yaxis2=dict(title='Dropout Rate (%)', overlaying='y', side='right'),
                    legend=dict(orientation='h'),
                    height=350,
                )
                st.plotly_chart(fig_trend, use_container_width=True)

        # Temporal dropout analysis
        st.markdown('<div class="section-header">⏱️ Dropout Timing</div>',
                    unsafe_allow_html=True)

        dropout_months = df_filtered[df_filtered['dropout_label'] == 1]['dropout_month'].dropna()
        fig_months = px.histogram(
            dropout_months, x=dropout_months.values,
            nbins=12,
            title='Dropout Month Distribution (Critical window: months 2-4)',
            labels={'x': 'Treatment Month', 'y': 'N Dropouts'},
            color_discrete_sequence=['#E74C3C'],
        )
        fig_months.add_vrect(x0=2, x1=4, fillcolor='orange', opacity=0.15,
                              annotation_text='Critical window', annotation_position='top right')
        fig_months.update_layout(height=300)
        st.plotly_chart(fig_months, use_container_width=True)


# ════════════════════════════════════════════════════════════
# TAB 2: REGIONAL ANALYSIS
# ════════════════════════════════════════════════════════════
with tab2:
    if df_filtered.empty:
        st.warning("No data for the selected filters.")
    else:
        st.markdown('<div class="section-header">🗺️ Geographic and Demographic Analysis</div>',
                    unsafe_allow_html=True)

        region_stats = df_filtered.groupby('region').agg(
            n_patients=('patient_id', 'count'),
            dropout_rate=('dropout_label', 'mean'),
            avg_distance=('distance_km', 'mean'),
            hiv_rate=('hiv_positive', 'mean'),
            avg_attendance=('attendance_rate_pct', 'mean'),
            avg_age=('age', 'mean'),
        ).reset_index()
        region_stats['dropout_pct'] = (region_stats['dropout_rate'] * 100).round(1)
        region_stats['hiv_pct'] = (region_stats['hiv_rate'] * 100).round(1)

        col1, col2 = st.columns(2)

        with col1:
            fig_reg = px.bar(
                region_stats.sort_values('dropout_pct', ascending=True),
                x='dropout_pct', y='region',
                orientation='h',
                color='dropout_pct',
                color_continuous_scale='RdYlGn_r',
                text='dropout_pct',
                title='Dropout Rate by Region (%)',
                labels={'dropout_pct': 'Dropout (%)', 'region': 'Region'},
            )
            fig_reg.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            fig_reg.update_layout(height=350, showlegend=False)
            st.plotly_chart(fig_reg, use_container_width=True)

        with col2:
            fig_scatter = px.scatter(
                region_stats,
                x='avg_distance', y='dropout_pct',
                size='n_patients',
                color='hiv_pct',
                color_continuous_scale='Oranges',
                text='region',
                title='Distance to Clinic vs. Dropout Rate',
                labels={
                    'avg_distance': 'Mean distance (km)',
                    'dropout_pct': 'Dropout (%)',
                    'hiv_pct': '% HIV+'
                },
                hover_data=['avg_attendance', 'avg_age'],
            )
            fig_scatter.update_traces(textposition='top center')
            fig_scatter.update_layout(height=350)
            st.plotly_chart(fig_scatter, use_container_width=True)

        # Regional summary table
        st.markdown('<div class="section-header">📋 Regional Summary Table</div>',
                    unsafe_allow_html=True)
        display_cols = {
            'region': 'Region', 'n_patients': 'N Patients',
            'dropout_pct': 'Dropout (%)', 'avg_distance': 'Mean Dist. (km)',
            'hiv_pct': 'HIV+ (%)', 'avg_attendance': 'Mean Attendance (%)',
            'avg_age': 'Mean Age'
        }
        df_display = region_stats[list(display_cols.keys())].rename(columns=display_cols)
        df_display = df_display.sort_values('Dropout (%)', ascending=False)

        st.dataframe(
            df_display,
            use_container_width=True,
            column_config={
                'Dropout (%)': st.column_config.ProgressColumn(
                    min_value=0, max_value=40, format="%.1f%%"
                ),
                'Mean Attendance (%)': st.column_config.ProgressColumn(
                    min_value=0, max_value=100, format="%.1f%%"
                ),
            }
        )

        # Age group analysis
        st.markdown('<div class="section-header">👥 Dropout by Age Group and Sex</div>',
                    unsafe_allow_html=True)

        if 'age_group' in df_filtered.columns:
            col_age1, col_age2 = st.columns(2)
            with col_age1:
                age_drop = df_filtered.groupby('age_group')['dropout_label'].mean().reset_index()
                age_drop['pct'] = age_drop['dropout_label'] * 100
                fig_age = px.bar(age_drop, x='age_group', y='pct',
                                  color='pct', color_continuous_scale='RdYlGn_r',
                                  title='Dropout by Age Group',
                                  labels={'age_group': 'Age Group', 'pct': 'Dropout (%)'})
                fig_age.update_layout(height=280, showlegend=False)
                st.plotly_chart(fig_age, use_container_width=True)

            with col_age2:
                sex_hiv = df_filtered.groupby(['sex', 'hiv_status'])['dropout_label'].mean().reset_index()
                sex_hiv['pct'] = sex_hiv['dropout_label'] * 100
                fig_sex = px.bar(sex_hiv, x='sex', y='pct', color='hiv_status',
                                  barmode='group',
                                  color_discrete_map={'Negative': '#27AE60',
                                                       'Positive': '#E74C3C', 'Unknown': '#95A5A6'},
                                  title='Dropout by Sex and HIV Status',
                                  labels={'sex': 'Sex', 'pct': 'Dropout (%)'})
                fig_sex.update_layout(height=280)
                st.plotly_chart(fig_sex, use_container_width=True)


# ════════════════════════════════════════════════════════════
# TAB 3: HIGH-RISK PATIENTS
# ════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">🎯 Top 10 Highest-Risk Patients for Dropout</div>',
                unsafe_allow_html=True)
    st.markdown("*This table is updated with model predictions each time Notebook 04 is run.*")
    st.markdown("*Health workers: prioritize immediate contact with patients flagged in red.*")

    if top10 is not None:
        df_top_display = top10.copy()

        def risk_badge(tier):
            if tier == 'HIGH':   return '🔴 HIGH'
            if tier == 'MEDIUM': return '🟡 MEDIUM'
            return '🟢 LOW'

        if 'Riesgo' in df_top_display.columns:
            df_top_display['Risk'] = df_top_display['Riesgo'].apply(risk_badge)
            df_top_display = df_top_display.drop(columns=['Riesgo'])
        elif 'Risk' in df_top_display.columns:
            df_top_display['Risk'] = df_top_display['Risk'].apply(risk_badge)

        st.dataframe(
            df_top_display,
            use_container_width=True,
            column_config={
                'P(Dropout)': st.column_config.ProgressColumn(
                    'P(Dropout)', min_value=0, max_value=1, format="%.3f"
                ),
                'Top Risk Factor': st.column_config.TextColumn(
                    'Top Risk Factor (LIME)', width='large'
                ),
            },
            hide_index=True,
        )

        # Download action list
        csv_out = df_top_display.to_csv(index=False)
        st.download_button(
            label="⬇️ Export action list (CSV)",
            data=csv_out,
            file_name=f"high_risk_patients_{pd.Timestamp.today().date()}.csv",
            mime="text/csv",
        )
    else:
        st.warning("""
        ⚠️ No predictions available yet.

        Run Notebook 04 to generate predictions:
        ```bash
        jupyter nbconvert --to notebook --execute notebooks/04_explainability_lime.ipynb
        ```
        """)

    st.markdown("---")
    st.markdown('<div class="section-header">📊 Model Risk Score Distribution</div>',
                unsafe_allow_html=True)

    if model is not None and df is not None:
        from feature_engineering import prepare_ml_dataset
        X_all, y_all, _, _, _ = prepare_ml_dataset(df, encode_categoricals=True)
        y_prob_all = model.predict_proba(X_all)[:, 1]

        n_high   = (y_prob_all >= 0.7).sum()
        n_medium = ((y_prob_all >= 0.5) & (y_prob_all < 0.7)).sum()
        n_low    = (y_prob_all < 0.5).sum()

        col_r1, col_r2, col_r3 = st.columns(3)
        col_r1.metric("🔴 High Risk", f"{n_high}", f"{n_high/len(y_prob_all)*100:.1f}%")
        col_r2.metric("🟡 Medium Risk", f"{n_medium}", f"{n_medium/len(y_prob_all)*100:.1f}%")
        col_r3.metric("🟢 Low Risk", f"{n_low}", f"{n_low/len(y_prob_all)*100:.1f}%")

        fig_hist = px.histogram(
            x=y_prob_all, nbins=30,
            color_discrete_sequence=['#2C3E50'],
            title=f'Dropout Probability Distribution (n={len(y_prob_all):,})',
            labels={'x': 'P(Dropout)', 'y': 'N Patients'},
        )
        fig_hist.add_vline(x=threshold_adj, line_dash='dash', line_color='red',
                            annotation_text=f'Threshold={threshold_adj:.2f}')
        fig_hist.update_layout(height=300)
        st.plotly_chart(fig_hist, use_container_width=True)


# ════════════════════════════════════════════════════════════
# TAB 4: INDIVIDUAL PREDICTION
# ════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-header">🔍 Risk Prediction for New Patient</div>',
                unsafe_allow_html=True)
    st.markdown("Enter patient data to get an instant risk assessment.")

    if model is None:
        st.error("⚠️ Model not available. Run Notebook 03 first.")
    else:
        col_form1, col_form2, col_form3 = st.columns(3)

        with col_form1:
            st.markdown("**🧑 Demographics**")
            age_inp    = st.number_input("Age", 15, 80, 35)
            sex_inp    = st.selectbox("Sex", ['Male', 'Female'])
            region_inp = st.selectbox("Region", ['Norte', 'Sur', 'Este', 'Oeste', 'Centro'])
            edu_inp    = st.selectbox("Education", ['None', 'Primary', 'Secondary', 'Tertiary'])
            emp_inp    = st.selectbox("Employment", ['Employed', 'Unemployed', 'Informal', 'Student'])

        with col_form2:
            st.markdown("**🏥 Clinical**")
            hiv_inp    = st.selectbox("HIV Status", ['Negative', 'Positive', 'Unknown'])
            alcohol_inp = st.selectbox("Alcohol use", ['None', 'Occasional', 'Heavy'])
            smoking_inp = st.selectbox("Smoking", ['Never', 'Former', 'Current'])
            prev_tb_inp = st.checkbox("Previous TB")
            diabetes_inp = st.checkbox("Diabetes")
            resist_inp  = st.selectbox("Drug resistance", ['Sensitive', 'MDR', 'XDR', 'Unknown'])
            tb_type_inp = st.selectbox("TB type", ['Pulmonary', 'Extrapulmonary', 'Both'])

        with col_form3:
            st.markdown("**📍 Logistical**")
            dist_inp   = st.number_input("Distance to clinic (km)", 0.0, 200.0, 15.0, 1.0)
            contacts_inp = st.number_input("Household contacts", 0, 15, 3)
            regimen_inp = st.selectbox("Regimen", ['2HRZE/4HR', '2HRZE/4HR3', 'HRZE/HRE', 'BPaL'])
            supporter_inp = st.selectbox("DOT support", ['Family', 'CHW', 'Clinic', 'None'])
            dot_inp     = st.selectbox("DOT method", ['In-person', 'Video DOT', 'Self-administered'])
            visits_inp  = st.number_input("Visits attended so far", 0, 24, 3)
            total_visits_inp = st.number_input("Total visits scheduled so far", 1, 24, 4)

        if st.button("🔍 Assess Risk", type="primary", use_container_width=True):
            # Build feature vector
            att_rate = visits_inp / total_visits_inp * 100 if total_visits_inp > 0 else 0
            missed = total_visits_inp - visits_inp
            expected_months = 6 if '4HR' in regimen_inp else 8 if regimen_inp == 'HRZE/HRE' else 18

            manual_features = {
                'age': age_inp,
                'distance_km': dist_inp,
                'household_contacts': contacts_inp,
                'total_visits': total_visits_inp,
                'visits_attended': visits_inp,
                'visits_missed': missed,
                'attendance_rate_pct': att_rate,
                'avg_adherence': att_rate,
                'n_severe_side_effects': 0,
                'n_social_worker': 0,
                'weight_gain_kg': 1.0,
                'expected_months': expected_months,
                'diabetes': int(diabetes_inp),
                'previous_tb': int(prev_tb_inp),
                'hiv_positive': int(hiv_inp == 'Positive'),
                'far_from_clinic': int(dist_inp > 20),
                'very_far': int(dist_inp > 40),
                'hiv_and_far': int(hiv_inp == 'Positive' and dist_inp > 20),
                'low_attendance': int(att_rate < 70),
                'smear_positive_m2': 0,
                'high_toxicity': 0,
                'heavy_alcohol': int(alcohol_inp == 'Heavy'),
                'current_smoker': int(smoking_inp == 'Current'),
                'unemployed': int(emp_inp == 'Unemployed'),
                'no_education': int(edu_inp == 'None'),
                'mdr_xdr': int(resist_inp in ['MDR', 'XDR']),
                'support_score': (
                    3 * (supporter_inp == 'CHW') + 2 * (supporter_inp == 'Family') +
                    1 * (supporter_inp == 'Clinic') + 2 * (dot_inp == 'In-person')
                ),
            }

            # Categorical encodings
            cat_map = {
                'sex': sex_inp, 'region': region_inp, 'education_level': edu_inp,
                'employment_status': emp_inp, 'hiv_status': hiv_inp,
                'alcohol_use': alcohol_inp, 'smoking': smoking_inp,
                'tb_type': tb_type_inp, 'drug_resistance': resist_inp,
                'regimen': regimen_inp, 'supporter': supporter_inp,
                'dot_method': dot_inp, 'age_group': '25-35',
            }

            for cat, val in cat_map.items():
                key = cat + '_enc'
                if encoders and cat in encoders:
                    try:
                        encoded = encoders[cat].transform([val])[0]
                    except ValueError:
                        encoded = 0
                    manual_features[key] = encoded
                else:
                    manual_features[key] = 0

            x_vec = np.array([manual_features.get(f, 0) for f in feature_names]).reshape(1, -1)
            prob = model.predict_proba(x_vec)[0, 1]

            st.markdown("---")

            col_res1, col_res2, col_res3 = st.columns(3)

            with col_res1:
                if prob >= 0.7:
                    st.error(f"## 🔴 HIGH RISK\n### P(dropout) = {prob:.3f}")
                    st.markdown("**Recommended action:** Immediate social worker contact")
                elif prob >= 0.5:
                    st.warning(f"## 🟡 MEDIUM RISK\n### P(dropout) = {prob:.3f}")
                    st.markdown("**Recommended action:** Follow-up call within 1 week")
                else:
                    st.success(f"## 🟢 LOW RISK\n### P(dropout) = {prob:.3f}")
                    st.markdown("**Action:** Routine follow-up")

            with col_res2:
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=prob * 100,
                    number={'suffix': '%', 'font': {'size': 28}},
                    title={'text': "Dropout Probability"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': '#E74C3C' if prob >= 0.7
                                else '#F39C12' if prob >= 0.5 else '#27AE60'},
                        'steps': [
                            {'range': [0, 50], 'color': '#EAFAF1'},
                            {'range': [50, 70], 'color': '#FEF9E7'},
                            {'range': [70, 100], 'color': '#FDEDEC'},
                        ],
                        'threshold': {
                            'line': {'color': 'black', 'width': 2},
                            'thickness': 0.75,
                            'value': threshold_adj * 100
                        },
                    }
                ))
                fig_gauge.update_layout(height=250)
                st.plotly_chart(fig_gauge, use_container_width=True)

            with col_res3:
                st.markdown("**📊 Detected risk factors:**")
                risk_factors = []
                if dist_inp > 20:
                    risk_factors.append(f"📍 High distance: {dist_inp:.0f} km")
                if hiv_inp == 'Positive':
                    risk_factors.append("🔴 HIV co-infection")
                if alcohol_inp == 'Heavy':
                    risk_factors.append("🍺 Heavy alcohol use")
                if att_rate < 70:
                    risk_factors.append(f"📅 Low attendance: {att_rate:.0f}%")
                if resist_inp in ('MDR', 'XDR'):
                    risk_factors.append(f"💊 Drug resistance: {resist_inp}")
                if prev_tb_inp:
                    risk_factors.append("🔄 Previous TB (relapse)")
                if supporter_inp == 'None':
                    risk_factors.append("👤 No DOT supporter assigned")
                if edu_inp == 'None':
                    risk_factors.append("📚 No formal education")

                if risk_factors:
                    for rf in risk_factors[:5]:
                        st.markdown(f"- {rf}")
                else:
                    st.markdown("- ✅ No significant risk factors detected")

# FOOTER
st.markdown("---")
st.markdown(
    "<center><small>TB Churn Dashboard · Facundo Colaccini PhD · "
    "Random Forest + LIME · Streamlit</small></center>",
    unsafe_allow_html=True
)
