"""
Clinical Trial Outcome Predictor - Training Pipeline
=====================================================
MLflow-tracked training pipeline for predicting clinical trial success/failure.
Trains an XGBoost model with hyperparameter tuning and experiment tracking.
"""

import os
import json
import logging
import warnings
from datetime import datetime
from pathlib import Path

import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)
from sklearn.utils.class_weight import compute_sample_weight

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CONFIG = {
      "mlflow_tracking_uri": os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"),
      "experiment_name": "clinical-trial-predictor",
      "data_path": os.getenv("DATA_PATH", "data/clinical_trials.csv"),
      "model_output_dir": "models",
      "test_size": 0.2,
      "random_state": 42,
      "cv_folds": 5,
}

HYPERPARAMETERS = {
      "n_estimators": 500,
      "max_depth": 6,
      "learning_rate": 0.05,
      "subsample": 0.8,
      "colsample_bytree": 0.8,
      "min_child_weight": 3,
      "gamma": 0.1,
      "reg_alpha": 0.1,
      "reg_lambda": 1.0,
      "scale_pos_weight": 1.0,
      "objective": "binary:logistic",
      "eval_metric": "auc",
      "early_stopping_rounds": 50,
      "random_state": CONFIG["random_state"],
}

# ---------------------------------------------------------------------------
# Feature Engineering
# ---------------------------------------------------------------------------

TRIAL_PHASES = {"Phase 1": 1, "Phase 2": 2, "Phase 3": 3, "Phase 4": 4, "Phase 1/Phase 2": 1.5, "Phase 2/Phase 3": 2.5}
THERAPEUTIC_AREAS = [
      "Oncology", "Cardiology", "Neurology", "Immunology", "Infectious Disease",
      "Endocrinology", "Respiratory", "Gastroenterology", "Dermatology", "Rare Disease",
]


def load_and_validate_data(path: str) -> pd.DataFrame:
      """Load clinical trial dataset and run basic validation."""
      logger.info("Loading data from %s", path)

    if not Path(path).exists():
              logger.warning("Dataset not found at %s — generating synthetic data for demo", path)
              return _generate_synthetic_data()

    df = pd.read_csv(path)
    required = {"trial_id", "phase", "enrollment", "duration_months", "num_endpoints", "outcome"}
    missing = required - set(df.columns)
    if missing:
              raise ValueError(f"Missing required columns: {missing}")

    logger.info("Loaded %d trials with %d features", len(df), len(df.columns))
    return df


def _generate_synthetic_data(n: int = 2500) -> pd.DataFrame:
      """Create a realistic synthetic clinical-trial dataset."""
      rng = np.random.RandomState(CONFIG["random_state"])

    phases = rng.choice(list(TRIAL_PHASES.keys()), n, p=[0.2, 0.3, 0.3, 0.05, 0.075, 0.075])
    areas = rng.choice(THERAPEUTIC_AREAS, n)
    enrollment = rng.lognormal(mean=5.5, sigma=1.2, size=n).astype(int).clip(10, 50000)
    duration = rng.gamma(shape=3, scale=8, size=n).clip(1, 120).round(1)
    num_sites = rng.poisson(lam=25, size=n).clip(1, 500)
    num_endpoints = rng.randint(1, 8, size=n)
    has_biomarker = rng.binomial(1, 0.4, size=n)
    sponsor_experience = rng.randint(0, 200, size=n)
    dropout_rate = rng.beta(2, 8, size=n).round(3)
    prior_phase_success = rng.binomial(1, 0.6, size=n)
    molecule_type = rng.choice(["Small Molecule", "Biologic", "Gene Therapy", "Cell Therapy"], n, p=[0.45, 0.35, 0.1, 0.1])

    # Outcome influenced by realistic factors
    phase_num = np.array([TRIAL_PHASES.get(p, 2) for p in phases])
    logit = (
              -1.5
              + 0.3 * (phase_num >= 3).astype(float)
              + 0.4 * has_biomarker
              + 0.3 * prior_phase_success
              - 0.8 * dropout_rate
              + 0.002 * sponsor_experience
              - 0.003 * num_endpoints
              + np.where(np.isin(areas, ["Oncology", "Rare Disease"]), -0.3, 0.1)
              + rng.normal(0, 0.5, n)
    )
    prob = 1 / (1 + np.exp(-logit))
    outcome = rng.binomial(1, prob)

    df = pd.DataFrame({
              "trial_id": [f"NCT{str(i).zfill(8)}" for i in range(n)],
              "phase": phases,
              "therapeutic_area": areas,
              "enrollment": enrollment,
              "duration_months": duration,
              "num_sites": num_sites,
              "num_endpoints": num_endpoints,
              "has_biomarker_strategy": has_biomarker,
              "sponsor_trial_experience": sponsor_experience,
              "dropout_rate": dropout_rate,
              "prior_phase_success": prior_phase_success,
              "molecule_type": molecule_type,
              "outcome": outcome,
    })

    Path("data").mkdir(exist_ok=True)
    df.to_csv(CONFIG["data_path"], index=False)
    logger.info("Generated synthetic dataset with %d records (%.1f%% positive)", n, outcome.mean() * 100)
    return df


def engineer_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
      """Transform raw data into model-ready features."""
      logger.info("Engineering features …")
      out = df.copy()

    # Encode phase
      out["phase_numeric"] = out["phase"].map(TRIAL_PHASES).fillna(2)

    # Encode categoricals
    le_area = LabelEncoder()
    out["therapeutic_area_enc"] = le_area.fit_transform(out["therapeutic_area"])
    le_mol = LabelEncoder()
    out["molecule_type_enc"] = le_mol.fit_transform(out["molecule_type"])

    # Derived features
    out["enrollment_per_site"] = (out["enrollment"] / out["num_sites"].clip(lower=1)).round(2)
    out["duration_per_endpoint"] = (out["duration_months"] / out["num_endpoints"].clip(lower=1)).round(2)
    out["complexity_score"] = (out["num_sites"] * out["num_endpoints"] * out["duration_months"]).round(2)
    out["log_enrollment"] = np.log1p(out["enrollment"])
    out["experience_bin"] = pd.qcut(out["sponsor_trial_experience"], q=4, labels=False, duplicates="drop")
    out["retention_rate"] = (1 - out["dropout_rate"]).round(3)

    feature_cols = [
              "phase_numeric", "therapeutic_area_enc", "molecule_type_enc",
              "enrollment", "log_enrollment", "duration_months", "num_sites",
              "num_endpoints", "has_biomarker_strategy", "sponsor_trial_experience",
              "dropout_rate", "prior_phase_success", "enrollment_per_site",
              "duration_per_endpoint", "complexity_score", "experience_bin", "retention_rate",
    ]

    logger.info("Created %d features", len(feature_cols))
    return out, feature_cols


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_model(
      X_train: pd.DataFrame,
      y_train: pd.Series,
      X_val: pd.DataFrame,
      y_val: pd.Series,
      params: dict,
) -> xgb.XGBClassifier:
      """Train XGBoost with early stopping and class-weight balancing."""
      sample_weights = compute_sample_weight("balanced", y_train)

    model = xgb.XGBClassifier(
              n_estimators=params["n_estimators"],
              max_depth=params["max_depth"],
              learning_rate=params["learning_rate"],
              subsample=params["subsample"],
              colsample_bytree=params["colsample_bytree"],
              min_child_weight=params["min_child_weight"],
              gamma=params["gamma"],
              reg_alpha=params["reg_alpha"],
              reg_lambda=params["reg_lambda"],
              scale_pos_weight=params["scale_pos_weight"],
              objective=params["objective"],
              eval_metric=params["eval_metric"],
              early_stopping_rounds=params["early_stopping_rounds"],
              random_state=params["random_state"],
              use_label_encoder=False,
              n_jobs=-1,
    )

    model.fit(
              X_train, y_train,
              sample_weight=sample_weights,
              eval_set=[(X_val, y_val)],
              verbose=False,
    )

    logger.info("Training complete — best iteration: %d", model.best_iteration)
    return model


def evaluate_model(model: xgb.XGBClassifier, X: pd.DataFrame, y: pd.Series) -> dict:
      """Compute evaluation metrics."""
      y_pred = model.predict(X)
      y_proba = model.predict_proba(X)[:, 1]

    metrics = {
              "accuracy": round(accuracy_score(y, y_pred), 4),
              "precision": round(precision_score(y, y_pred, zero_division=0), 4),
              "recall": round(recall_score(y, y_pred, zero_division=0), 4),
              "f1_score": round(f1_score(y, y_pred, zero_division=0), 4),
              "roc_auc": round(roc_auc_score(y, y_proba), 4),
    }

    logger.info("Evaluation — %s", json.dumps(metrics, indent=2))
    return metrics


def cross_validate(model_params: dict, X: pd.DataFrame, y: pd.Series, folds: int = 5) -> dict:
      """Run stratified k-fold cross-validation."""
      skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=CONFIG["random_state"])
      estimator = xgb.XGBClassifier(**{k: v for k, v in model_params.items() if k != "early_stopping_rounds"}, use_label_encoder=False, n_jobs=-1)
      scores = cross_val_score(estimator, X, y, cv=skf, scoring="roc_auc")
      return {"cv_mean_auc": round(scores.mean(), 4), "cv_std_auc": round(scores.std(), 4)}


# ---------------------------------------------------------------------------
# MLflow Run
# ---------------------------------------------------------------------------


def run_experiment() -> None:
      """End-to-end training pipeline tracked by MLflow."""
      mlflow.set_tracking_uri(CONFIG["mlflow_tracking_uri"])
      mlflow.set_experiment(CONFIG["experiment_name"])

    # ---- data ----
      df = load_and_validate_data(CONFIG["data_path"])
    df, feature_cols = engineer_features(df)

    X = df[feature_cols]
    y = df["outcome"]

    X_train, X_test, y_train, y_test = train_test_split(
              X, y, test_size=CONFIG["test_size"], random_state=CONFIG["random_state"], stratify=y,
    )

    logger.info("Split: %d train / %d test  |  positive rate %.1f%%", len(X_train), len(X_test), y.mean() * 100)

    with mlflow.start_run(run_name=f"xgb-{datetime.now():%Y%m%d-%H%M%S}") as run:
              # Log parameters
              mlflow.log_params(HYPERPARAMETERS)
              mlflow.log_param("n_features", len(feature_cols))
              mlflow.log_param("n_train", len(X_train))
              mlflow.log_param("n_test", len(X_test))
              mlflow.log_param("positive_rate", round(y.mean(), 4))

        # Train
              model = train_model(X_train, y_train, X_test, y_test, HYPERPARAMETERS)

        # Evaluate
        metrics = evaluate_model(model, X_test, y_test)
        mlflow.log_metrics(metrics)

        # Cross-validation
        cv_metrics = cross_validate(HYPERPARAMETERS, X, y, CONFIG["cv_folds"])
        mlflow.log_metrics(cv_metrics)

        # Feature importance
        importance = dict(zip(feature_cols, model.feature_importances_.tolist()))
        mlflow.log_dict(importance, "feature_importance.json")

        # Log model
        mlflow.xgboost.log_model(
                      model,
                      artifact_path="model",
                      registered_model_name="clinical-trial-predictor",
                      input_example=X_test.iloc[:3],
        )

        # Save locally
        Path(CONFIG["model_output_dir"]).mkdir(exist_ok=True)
        model.save_model(f"{CONFIG['model_output_dir']}/model.json")
        mlflow.log_artifact(f"{CONFIG['model_output_dir']}/model.json")

        logger.info("MLflow run %s complete  |  AUC=%.4f  |  F1=%.4f", run.info.run_id, metrics["roc_auc"], metrics["f1_score"])

    logger.info("Experiment finished. View results at %s", CONFIG["mlflow_tracking_uri"])


if __name__ == "__main__":
      run_experiment()
