"""
Clinical Trial Outcome Predictor - Streamlit Demo
==================================================
Interactive dashboard for exploring model predictions, feature importance,
and clinical trial analytics powered by the MLOps pipeline.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import json

# ---------------------------------------------------------------------------
# Page Config
# ---------------------------------------------------------------------------

st.set_page_config(
      page_title="Clinical Trial Predictor",
      page_icon="🧬",
      layout="wide",
      initial_sidebar_state="expanded",
)

API_URL = "http://localhost:8000"

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.title("🧬 Clinical Trial Predictor")
st.sidebar.markdown("---")
page = st.sidebar.radio(
      "Navigation",
      ["🏠 Overview", "🔮 Predict Outcome", "📊 Model Analytics", "📈 Portfolio Explorer"],
)
st.sidebar.markdown("---")
st.sidebar.markdown("**MLOps Pipeline Demo**")
st.sidebar.caption("Built with FastAPI · MLflow · XGBoost · Streamlit")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PHASES = ["Phase 1", "Phase 1/Phase 2", "Phase 2", "Phase 2/Phase 3", "Phase 3", "Phase 4"]
THERAPEUTIC_AREAS = [
      "Oncology", "Cardiology", "Neurology", "Immunology", "Infectious Disease",
      "Endocrinology", "Respiratory", "Gastroenterology", "Dermatology", "Rare Disease",
]
MOLECULE_TYPES = ["Small Molecule", "Biologic", "Gene Therapy", "Cell Therapy"]

# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------


def check_api_health() -> bool:
      """Check if the FastAPI backend is available."""
      try:
                resp = requests.get(f"{API_URL}/health", timeout=3)
                return resp.status_code == 200
except requests.ConnectionError:
        return False


def predict_trial(payload: dict) -> dict | None:
      """Send prediction request to the API."""
      try:
                resp = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
                resp.raise_for_status()
                return resp.json()
except Exception as e:
        st.error(f"Prediction failed: {e}")
        return None


def generate_demo_portfolio(n: int = 200) -> pd.DataFrame:
      """Generate a synthetic portfolio of clinical trials for analytics."""
      rng = np.random.RandomState(42)
      return pd.DataFrame({
          "trial_id": [f"NCT{str(i).zfill(8)}" for i in range(n)],
          "phase": rng.choice(PHASES, n),
          "therapeutic_area": rng.choice(THERAPEUTIC_AREAS, n),
          "molecule_type": rng.choice(MOLECULE_TYPES, n),
          "enrollment": rng.lognormal(5.5, 1.0, n).astype(int).clip(20, 10000),
          "duration_months": rng.gamma(3, 8, n).clip(3, 96).round(1),
          "num_sites": rng.poisson(25, n).clip(1, 300),
          "num_endpoints": rng.randint(1, 7, n),
          "has_biomarker_strategy": rng.binomial(1, 0.4, n),
          "sponsor_trial_experience": rng.randint(5, 180, n),
          "dropout_rate": rng.beta(2, 8, n).round(3),
          "prior_phase_success": rng.binomial(1, 0.6, n),
          "predicted_success_prob": rng.beta(5, 5, n).round(3),
      })


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------


def overview_page():
      """Landing page with project overview and pipeline status."""
      st.title("🧬 Clinical Trial Outcome Predictor")
      st.markdown("### End-to-End MLOps Pipeline for Life Sciences")
      st.markdown("---")

    # Pipeline status
      col1, col2, col3, col4 = st.columns(4)
      api_healthy = check_api_health()

    col1.metric("API Status", "Online ✅" if api_healthy else "Offline ❌")
    col2.metric("Model", "XGBoost v1.0")
    col3.metric("Features", "17")
    col4.metric("Training AUC", "0.847")

    st.markdown("---")

    # Architecture
    st.subheader("Pipeline Architecture")
    st.markdown("""
        ```
            ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
                │  Data     │───▶│ Feature  │───▶│  Train   │───▶│  Deploy  │───▶│  Serve   │
                    │ Ingestion │    │ Engineer │    │ + Track  │    │ (Docker) │    │ (FastAPI)│
                        └──────────┘    └──────────┘    └──────────┘    └──────────┘    └──────────┘
                                 │                               │                               │
                                          ▼                               ▼                               ▼
                                              CSV / Database              MLflow Experiment              Streamlit Dashboard
                                                  ```
                                                      """)

    st.markdown("---")

    # Key features
    st.subheader("Key Capabilities")
    feat_col1, feat_col2, feat_col3 = st.columns(3)
    with feat_col1:
              st.markdown("**🔮 Predictive Analytics**")
              st.markdown("XGBoost model predicting clinical trial success probability based on 17 engineered features.")
          with feat_col2:
                    st.markdown("**📊 Experiment Tracking**")
                    st.markdown("Full MLflow integration for parameter logging, metric tracking, and model versioning.")
                with feat_col3:
                          st.markdown("**🚀 Production Deployment**")
                          st.markdown("Dockerized FastAPI serving with health checks, CI/CD, and monitoring endpoints.")


def prediction_page():
      """Interactive prediction form."""
      st.title("🔮 Predict Trial Outcome")
      st.markdown("Enter clinical trial parameters to get a success probability prediction.")
      st.markdown("---")

    with st.form("prediction_form"):
              col1, col2, col3 = st.columns(3)

        with col1:
                      st.subheader("Trial Design")
                      phase = st.selectbox("Phase", PHASES, index=2)
                      therapeutic_area = st.selectbox("Therapeutic Area", THERAPEUTIC_AREAS)
                      molecule_type = st.selectbox("Molecule Type", MOLECULE_TYPES)
                      num_endpoints = st.slider("Number of Endpoints", 1, 8, 3)

        with col2:
                      st.subheader("Operational")
                      enrollment = st.number_input("Target Enrollment", min_value=10, max_value=50000, value=500)
                      duration_months = st.slider("Duration (months)", 1.0, 120.0, 24.0, 0.5)
                      num_sites = st.number_input("Number of Sites", min_value=1, max_value=500, value=30)
                      dropout_rate = st.slider("Expected Dropout Rate", 0.0, 0.5, 0.12, 0.01)

        with col3:
                      st.subheader("Strategy")
                      has_biomarker = st.toggle("Biomarker Strategy", value=True)
                      prior_phase_success = st.toggle("Prior Phase Success", value=True)
                      sponsor_experience = st.slider("Sponsor Trial Experience", 0, 200, 50)

        submitted = st.form_submit_button("🔮 Predict Outcome", use_container_width=True)

    if submitted:
              payload = {
                            "phase": phase,
                            "therapeutic_area": therapeutic_area,
                            "molecule_type": molecule_type,
                            "enrollment": enrollment,
                            "duration_months": duration_months,
                            "num_sites": num_sites,
                            "num_endpoints": num_endpoints,
                            "has_biomarker_strategy": int(has_biomarker),
                            "sponsor_trial_experience": sponsor_experience,
                            "dropout_rate": dropout_rate,
                            "prior_phase_success": int(prior_phase_success),
              }

        with st.spinner("Running prediction..."):
                      result = predict_trial(payload)

        if result:
                      st.markdown("---")
                      st.subheader("Prediction Results")

            res_col1, res_col2, res_col3 = st.columns(3)
            prob = result.get("success_probability", 0.5)
            outcome = result.get("predicted_outcome", "Unknown")

            res_col1.metric("Success Probability", f"{prob:.1%}")
            res_col2.metric("Predicted Outcome", outcome)
            res_col3.metric("Confidence", f"{abs(prob - 0.5) * 2:.1%}")

            # Gauge chart
            fig = go.Figure(go.Indicator(
                              mode="gauge+number+delta",
                              value=prob * 100,
                              title={"text": "Success Probability (%)"},
                              gauge={
                                                    "axis": {"range": [0, 100]},
                                                    "bar": {"color": "#2ecc71" if prob >= 0.5 else "#e74c3c"},
                                                    "steps": [
                                                                              {"range": [0, 30], "color": "#ffebee"},
                                                                              {"range": [30, 60], "color": "#fff3e0"},
                                                                              {"range": [60, 100], "color": "#e8f5e9"},
                                                    ],
                              },
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
else:
            st.warning("API is offline. Start the Docker containers to enable live predictions.")
              st.code("docker-compose up -d", language="bash")


def analytics_page():
      """Model performance and feature importance analytics."""
    st.title("📊 Model Analytics")
    st.markdown("---")

    # Simulated metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Accuracy", "0.792")
    col2.metric("Precision", "0.814")
    col3.metric("Recall", "0.756")
    col4.metric("F1 Score", "0.784")
    col5.metric("ROC AUC", "0.847")

    st.markdown("---")

    # Feature importance
    st.subheader("Feature Importance")
    features = {
              "dropout_rate": 0.142, "prior_phase_success": 0.128, "has_biomarker_strategy": 0.115,
              "sponsor_trial_experience": 0.098, "enrollment": 0.089, "phase_numeric": 0.082,
              "duration_months": 0.075, "num_endpoints": 0.068, "num_sites": 0.055,
              "complexity_score": 0.042, "retention_rate": 0.038, "log_enrollment": 0.025,
              "enrollment_per_site": 0.018, "therapeutic_area_enc": 0.012,
              "molecule_type_enc": 0.008, "duration_per_endpoint": 0.005,
    }

    fig_imp = px.bar(
              x=list(features.values()),
              y=list(features.keys()),
              orientation="h",
              title="XGBoost Feature Importance (Gain)",
              labels={"x": "Importance", "y": "Feature"},
              color=list(features.values()),
              color_continuous_scale="Viridis",
    )
    fig_imp.update_layout(height=500, showlegend=False, yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig_imp, use_container_width=True)

    # ROC curve (simulated)
    st.subheader("ROC Curve")
    fpr = np.array([0, 0.02, 0.05, 0.1, 0.15, 0.25, 0.4, 0.6, 0.8, 1.0])
    tpr = np.array([0, 0.25, 0.48, 0.65, 0.74, 0.85, 0.92, 0.96, 0.99, 1.0])

    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name="Model (AUC = 0.847)", line=dict(color="#2ecc71", width=2)))
    fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random", line=dict(color="gray", dash="dash")))
    fig_roc.update_layout(title="Receiver Operating Characteristic", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate", height=400)
    st.plotly_chart(fig_roc, use_container_width=True)


def portfolio_page():
      """Trial portfolio exploration dashboard."""
    st.title("📈 Portfolio Explorer")
    st.markdown("Analyze predicted outcomes across your clinical trial portfolio.")
    st.markdown("---")

    df = generate_demo_portfolio()

    # Filters
    filter_col1, filter_col2, filter_col3 = st.columns(3)
    with filter_col1:
              selected_phases = st.multiselect("Phase", PHASES, default=PHASES)
          with filter_col2:
                    selected_areas = st.multiselect("Therapeutic Area", THERAPEUTIC_AREAS, default=THERAPEUTIC_AREAS)
                with filter_col3:
                          prob_range = st.slider("Success Probability Range", 0.0, 1.0, (0.0, 1.0), 0.05)

    filtered = df[
        (df["phase"].isin(selected_phases))
        & (df["therapeutic_area"].isin(selected_areas))
        & (df["predicted_success_prob"].between(*prob_range))
]

    # KPIs
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Total Trials", len(filtered))
    kpi2.metric("Avg Success Prob", f"{filtered['predicted_success_prob'].mean():.1%}")
    kpi3.metric("Total Enrollment", f"{filtered['enrollment'].sum():,}")
    kpi4.metric("High Confidence", len(filtered[filtered["predicted_success_prob"] > 0.7]))

    st.markdown("---")

    # Charts
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
              fig1 = px.box(
                            filtered, x="phase", y="predicted_success_prob",
                            color="phase", title="Success Probability by Phase",
                            labels={"predicted_success_prob": "Success Probability"},
              )
              fig1.update_layout(showlegend=False, height=400)
              st.plotly_chart(fig1, use_container_width=True)

    with chart_col2:
              area_avg = filtered.groupby("therapeutic_area")["predicted_success_prob"].mean().sort_values()
              fig2 = px.bar(
                  x=area_avg.values, y=area_avg.index,
                  orientation="h", title="Avg Success Probability by Therapeutic Area",
                  labels={"x": "Avg Probability", "y": ""},
                  color=area_avg.values, color_continuous_scale="RdYlGn",
              )
              fig2.update_layout(height=400, showlegend=False)
              st.plotly_chart(fig2, use_container_width=True)

    # Scatter
    fig3 = px.scatter(
              filtered, x="enrollment", y="predicted_success_prob",
              color="therapeutic_area", size="duration_months",
              hover_data=["trial_id", "phase", "molecule_type"],
              title="Enrollment vs Success Probability",
              labels={"enrollment": "Enrollment", "predicted_success_prob": "Success Probability"},
    )
    fig3.update_layout(height=500)
    st.plotly_chart(fig3, use_container_width=True)

    # Data table
    st.subheader("Trial Details")
    st.dataframe(filtered.sort_values("predicted_success_prob", ascending=False), use_container_width=True, height=400)


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

if page == "🏠 Overview":
      overview_page()
elif page == "🔮 Predict Outcome":
    prediction_page()
elif page == "📊 Model Analytics":
    analytics_page()
elif page == "📈 Portfolio Explorer":
    portfolio_page()
