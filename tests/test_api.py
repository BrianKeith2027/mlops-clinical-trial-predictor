"""Tests for the FastAPI clinical trial prediction service.

These cover the contract the Streamlit demo and any downstream consumer rely on:
the response shape, validation behaviour, and the guarantee that the service
never reports metrics or explanations it did not actually produce.
"""

import pytest
from fastapi.testclient import TestClient

from src.api.main import (
    TrialInput,
    app,
    classify_risk,
    encode_features,
    heuristic_risk_factors,
    mock_probability,
)

VALID_PAYLOAD = {
    "phase": "Phase III",
    "therapeutic_area": "Oncology",
    "sponsor_type": "Industry",
    "enrollment": 450,
    "trial_design": "Randomized Controlled",
    "endpoint_type": "Overall Survival",
    "num_sites": 85,
    "duration_months": 36,
    "has_biomarker": True,
    "prior_phase_success": True,
}


@pytest.fixture(name="client")
def client_fixture():
    """TestClient as a context manager so startup handlers run."""
    with TestClient(app) as test_client:
        yield test_client


def test_health_reports_status_and_model_state(client):
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "healthy"
    assert isinstance(body["model_loaded"], bool)


def test_model_info_does_not_invent_metrics(client):
    """If no metrics artefact exists, the API must return null, not a number."""
    response = client.get("/model-info")

    assert response.status_code == 200
    body = response.json()
    if body["metrics_source"] == "unavailable":
        assert body["auc_roc"] is None
        assert body["training_samples"] is None


def test_predict_returns_expected_contract(client):
    response = client.post("/predict", json=VALID_PAYLOAD)

    assert response.status_code == 200
    body = response.json()
    assert 0.0 <= body["success_probability"] <= 1.0
    assert body["risk_level"] in {"Low", "Medium", "High"}
    assert len(body["confidence_interval"]) == 2
    assert body["confidence_interval"][0] <= body["confidence_interval"][1]
    assert body["top_risk_factors"]
    assert body["explanation_method"] in {"shap", "heuristic-fallback"}


def test_predict_labels_output_when_no_model_is_loaded(client):
    """A prediction made without a trained model must say so."""
    health = client.get("/health").json()
    body = client.post("/predict", json=VALID_PAYLOAD).json()

    if not health["model_loaded"]:
        assert body["model_version"] == "mock-fallback"


@pytest.mark.parametrize("field,bad_value", [("enrollment", 0), ("num_sites", 0), ("duration_months", 0)])
def test_predict_rejects_non_positive_counts(client, field, bad_value):
    payload = dict(VALID_PAYLOAD)
    payload[field] = bad_value

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_predict_rejects_missing_required_field(client):
    payload = dict(VALID_PAYLOAD)
    del payload["phase"]

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_encode_features_preserves_order_and_shape():
    trial = TrialInput(**VALID_PAYLOAD)

    features = encode_features(trial)

    assert features.shape == (1, 10)
    assert features[0][3] == VALID_PAYLOAD["enrollment"]
    assert features[0][8] == 1  # has_biomarker


def test_encode_features_defaults_unknown_categories():
    payload = dict(VALID_PAYLOAD)
    payload["therapeutic_area"] = "Not A Real Area"
    payload["phase"] = "Phase VII"

    features = encode_features(TrialInput(**payload))

    assert features[0][0] == 2  # unknown phase falls back to Phase II
    assert features[0][1] == 8  # unknown area falls back to Other


@pytest.mark.parametrize(
    "proba,expected",
    [(0.95, "Low"), (0.70, "Low"), (0.699, "Medium"), (0.40, "Medium"), (0.399, "High"), (0.0, "High")],
)
def test_classify_risk_boundaries(proba, expected):
    assert classify_risk(proba) == expected


def test_mock_probability_is_deterministic_and_bounded():
    trial = TrialInput(**VALID_PAYLOAD)

    first = mock_probability(trial)
    second = mock_probability(trial)

    assert first == second
    assert 0.05 <= first <= 0.95


def test_heuristic_risk_factors_flag_oncology_as_negative():
    trial = TrialInput(**VALID_PAYLOAD)

    factors = {factor.feature: factor for factor in heuristic_risk_factors(trial)}

    assert factors["therapeutic_area"].direction == "negative"
    assert factors["has_biomarker"].direction == "positive"
