"""
FastAPI Application - Clinical Trial Outcome Predictor
=======================================================
Author: Brian Stratton
Description: Production REST API for serving clinical trial
             success predictions with SHAP explainability.
"""

import json
import logging
import os
from datetime import datetime, timezone
from typing import Optional

import joblib
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

API_VERSION = "1.2.0"

# -- App Initialization --------------------------------------
app = FastAPI(
    title="Clinical Trial Outcome Predictor",
    description="ML-powered API for predicting clinical trial success probability",
    version=API_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -- Pydantic Schemas ----------------------------------------
class TrialInput(BaseModel):
    """Input schema for a clinical trial prediction request."""

    phase: str = Field(..., description="Trial phase", examples=["Phase III"])
    therapeutic_area: str = Field(..., description="Disease area", examples=["Oncology"])
    sponsor_type: str = Field(..., description="Sponsor category", examples=["Industry"])
    enrollment: int = Field(..., ge=1, description="Target enrollment", examples=[450])
    trial_design: str = Field(..., description="Study design", examples=["Randomized Controlled"])
    endpoint_type: str = Field(..., description="Primary endpoint", examples=["Overall Survival"])
    num_sites: int = Field(1, ge=1, description="Number of sites", examples=[85])
    duration_months: int = Field(12, ge=1, description="Planned duration", examples=[36])
    has_biomarker: bool = Field(False, description="Biomarker-driven trial")
    prior_phase_success: bool = Field(False, description="Prior phase met endpoints")


class RiskFactor(BaseModel):
    feature: str
    impact: float
    direction: str


class PredictionResponse(BaseModel):
    trial_id: str
    success_probability: float
    risk_level: str
    confidence_interval: list[float]
    top_risk_factors: list[RiskFactor]
    explanation_method: str
    model_version: str
    timestamp: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str


class ModelInfoResponse(BaseModel):
    model_type: str
    model_loaded: bool
    metrics_source: str
    training_date: Optional[str] = None
    auc_roc: Optional[float] = None
    features: Optional[int] = None
    training_samples: Optional[int] = None


# -- Feature Encoding ----------------------------------------
PHASE_MAP = {"Phase I": 1, "Phase II": 2, "Phase III": 3, "Phase IV": 4}
THERAPEUTIC_MAP = {
    "Oncology": 0, "Cardiovascular": 1, "Neurology": 2,
    "Immunology": 3, "Infectious Disease": 4, "Rare Disease": 5,
    "Metabolic": 6, "Respiratory": 7, "Other": 8,
}
SPONSOR_MAP = {"Industry": 0, "Academic": 1, "Government": 2, "Collaborative": 3}
DESIGN_MAP = {
    "Randomized Controlled": 0, "Single Arm": 1,
    "Crossover": 2, "Adaptive": 3, "Platform": 4,
}
ENDPOINT_MAP = {
    "Overall Survival": 0, "Progression-Free Survival": 1,
    "Objective Response Rate": 2, "Patient-Reported Outcome": 3,
    "Biomarker": 4, "Safety": 5, "Other": 6,
}

FEATURE_NAMES = [
    "phase",
    "therapeutic_area",
    "sponsor_type",
    "enrollment",
    "trial_design",
    "endpoint_type",
    "num_sites",
    "duration_months",
    "has_biomarker",
    "prior_phase_success",
]

# -- Model Loading -------------------------------------------
MODEL_PATH = os.getenv("MODEL_PATH", "models/xgb_trial_predictor.joblib")
METRICS_PATH = os.getenv("METRICS_PATH", "models/metrics.json")

model = None
model_metrics: dict = {}
explainer = None


def _load_explainer(loaded_model):
    """Build a SHAP TreeExplainer for the loaded model, if shap is available."""
    try:
        import shap

        return shap.TreeExplainer(loaded_model)
    except Exception as exc:  # pragma: no cover - depends on optional dependency
        logger.warning("SHAP explainer unavailable, falling back to heuristics: %s", exc)
        return None


@app.on_event("startup")
async def load_model():
    """Load the trained model and its metrics on application startup."""
    global model, model_metrics, explainer

    if os.path.exists(MODEL_PATH):
        try:
            model = joblib.load(MODEL_PATH)
            logger.info("Model loaded from %s", MODEL_PATH)
            explainer = _load_explainer(model)
        except Exception as exc:
            logger.error("Error loading model from %s: %s", MODEL_PATH, exc)
            model = None
    else:
        logger.warning(
            "Model not found at %s. API will return clearly-labelled mock predictions.",
            MODEL_PATH,
        )

    if os.path.exists(METRICS_PATH):
        try:
            with open(METRICS_PATH, "r", encoding="utf-8") as handle:
                model_metrics = json.load(handle)
            logger.info("Model metrics loaded from %s", METRICS_PATH)
        except Exception as exc:
            logger.error("Error loading metrics from %s: %s", METRICS_PATH, exc)
            model_metrics = {}


def encode_features(trial: TrialInput) -> np.ndarray:
    """Encode trial input into a single-row feature vector."""
    return np.array([[
        PHASE_MAP.get(trial.phase, 2),
        THERAPEUTIC_MAP.get(trial.therapeutic_area, 8),
        SPONSOR_MAP.get(trial.sponsor_type, 0),
        trial.enrollment,
        DESIGN_MAP.get(trial.trial_design, 0),
        ENDPOINT_MAP.get(trial.endpoint_type, 6),
        trial.num_sites,
        trial.duration_months,
        int(trial.has_biomarker),
        int(trial.prior_phase_success),
    ]])


def shap_risk_factors(features: np.ndarray, top_n: int = 3) -> Optional[list[RiskFactor]]:
    """Return the top contributing features using SHAP values.

    Returns None when no explainer is available so the caller can fall back.
    """
    if explainer is None:
        return None

    try:
        values = explainer.shap_values(features)
        row = np.asarray(values)[0] if np.asarray(values).ndim > 1 else np.asarray(values)
        ranked = sorted(
            zip(FEATURE_NAMES, row),
            key=lambda pair: abs(float(pair[1])),
            reverse=True,
        )[:top_n]
        return [
            RiskFactor(
                feature=name,
                impact=round(float(value), 4),
                direction="positive" if float(value) >= 0 else "negative",
            )
            for name, value in ranked
        ]
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("SHAP explanation failed, falling back to heuristics: %s", exc)
        return None


def heuristic_risk_factors(trial: TrialInput) -> list[RiskFactor]:
    """Deterministic, clearly-labelled stand-in used when SHAP is unavailable."""
    oncology_penalty = -0.12 if trial.therapeutic_area == "Oncology" else 0.04
    return [
        RiskFactor(
            feature="therapeutic_area",
            impact=oncology_penalty,
            direction="negative" if oncology_penalty < 0 else "positive",
        ),
        RiskFactor(
            feature="enrollment",
            impact=round(0.01 * min(trial.enrollment / 500, 1), 3),
            direction="positive",
        ),
        RiskFactor(
            feature="has_biomarker",
            impact=round(0.09 * int(trial.has_biomarker), 3),
            direction="positive" if trial.has_biomarker else "neutral",
        ),
    ]


def mock_probability(trial: TrialInput) -> float:
    """Deterministic placeholder probability used when no model is loaded."""
    base = 0.5
    base += 0.05 * PHASE_MAP.get(trial.phase, 2)
    base += 0.03 * int(trial.has_biomarker)
    base += 0.05 * int(trial.prior_phase_success)
    base -= 0.02 * (THERAPEUTIC_MAP.get(trial.therapeutic_area, 4) == 0)
    base += 0.01 * min(trial.enrollment / 500, 1)
    return max(0.05, min(0.95, base))


def classify_risk(proba: float) -> str:
    if proba >= 0.7:
        return "Low"
    if proba >= 0.4:
        return "Medium"
    return "High"


# -- API Endpoints -------------------------------------------
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint for container orchestration."""
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None,
        version=API_VERSION,
    )


@app.get("/model-info", response_model=ModelInfoResponse)
async def model_info():
    """Return model metadata and performance metrics.

    Metrics are read from the artefact written by the training run rather than
    hard-coded, so the API never reports numbers that were not measured.
    """
    return ModelInfoResponse(
        model_type=model_metrics.get("model_type", "XGBoost"),
        model_loaded=model is not None,
        metrics_source=METRICS_PATH if model_metrics else "unavailable",
        training_date=model_metrics.get("training_date"),
        auc_roc=model_metrics.get("auc_roc"),
        features=model_metrics.get("features", len(FEATURE_NAMES)),
        training_samples=model_metrics.get("training_samples"),
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(trial: TrialInput):
    """Predict clinical trial success probability.

    Returns success probability, risk level, confidence interval and the top
    contributing features. The explanation_method field states whether those
    contributions came from SHAP or from the deterministic fallback.
    """
    features = encode_features(trial)
    timestamp = datetime.now(timezone.utc)
    trial_id = f"pred_{timestamp.strftime('%Y%m%d_%H%M%S')}"

    if model is not None:
        proba = float(model.predict_proba(features)[0][1])
        model_version = API_VERSION
    else:
        proba = mock_probability(trial)
        model_version = "mock-fallback"

    risk_level = classify_risk(proba)

    margin = 0.07
    ci = [round(max(0.0, proba - margin), 3), round(min(1.0, proba + margin), 3)]

    risk_factors = shap_risk_factors(features)
    explanation_method = "shap"
    if risk_factors is None:
        risk_factors = heuristic_risk_factors(trial)
        explanation_method = "heuristic-fallback"

    logger.info(
        "Prediction %s: probability=%.3f, risk=%s, source=%s",
        trial_id,
        proba,
        risk_level,
        model_version,
    )

    return PredictionResponse(
        trial_id=trial_id,
        success_probability=round(proba, 3),
        risk_level=risk_level,
        confidence_interval=ci,
        top_risk_factors=risk_factors,
        explanation_method=explanation_method,
        model_version=model_version,
        timestamp=timestamp.isoformat().replace("+00:00", "Z"),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
