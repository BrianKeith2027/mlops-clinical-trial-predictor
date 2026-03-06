# 🧬 MLOps Clinical Trial Predictor

An end-to-end **MLOps pipeline** for predicting clinical trial outcomes (success/failure) in the pharmaceutical and life sciences domain. Features **FastAPI** model serving, **MLflow** experiment tracking, **Docker** containerization, **GitHub Actions** CI/CD, and a **Streamlit** demo interface.

## 📋 Overview

Clinical trials are expensive, time-consuming, and have historically high failure rates — roughly 90% of drug candidates that enter Phase I never reach FDA approval. This project applies machine learning to clinical trial metadata (phase, therapeutic area, sponsor type, enrollment size, trial design, endpoint type) to predict the probability of trial success, enabling pharmaceutical teams to prioritize resources toward higher-probability programs.

**End Use:** Pharmaceutical data science teams and clinical operations analysts use this tool to score incoming trial protocols, flag high-risk studies early, and support portfolio-level investment decisions — all served through a production REST API or an interactive Streamlit dashboard.

## 🎯 Key Features

| Feature | Description |
|---------|-------------|
| **XGBoost Classifier** | Gradient-boosted model trained on clinical trial features with SHAP explainability |
| **FastAPI Serving** | Production REST API with `/predict`, `/health`, and `/model-info` endpoints |
| **MLflow Tracking** | Full experiment tracking — parameters, metrics, artifacts, and model registry |
| **Docker Deployment** | Multi-service `docker-compose` stack (API + MLflow + Streamlit) |
| **GitHub Actions CI/CD** | Automated testing, linting, and Docker image builds on every push |
| **Streamlit Demo** | Interactive front-end for non-technical stakeholders to explore predictions |
| **SHAP Explainability** | Feature importance and force plots for every prediction |
| **Data Validation** | Pydantic schemas + Great Expectations data quality checks |

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        MLOps PIPELINE                                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐              │
│  │  📥 DATA      │    │  🧠 TRAIN     │    │  📦 REGISTRY  │              │
│  │              │    │              │    │              │              │
│  │  Ingest &    │───▶│  XGBoost +   │───▶│  MLflow      │              │
│  │  Validate    │    │  HPO Tuning  │    │  Model Store │              │
│  │              │    │              │    │              │              │
│  └──────────────┘    └──────────────┘    └──────┬───────┘              │
│                                                  │                      │
│                                                  ▼                      │
│                                        ┌──────────────────┐            │
│                                        │  🚀 SERVE         │            │
│                                        │  FastAPI + Docker │            │
│                                        └────────┬─────────┘            │
│                                                  │                      │
│                              ┌───────────────────┼───────────────┐     │
│                              ▼                   ▼               ▼     │
│                     ┌──────────────┐   ┌──────────────┐ ┌────────────┐│
│                     │  /predict    │   │  /health     │ │ Streamlit  ││
│                     │  REST API    │   │  Monitoring  │ │ Demo UI    ││
│                     └──────────────┘   └──────────────┘ └────────────┘│
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │  🔄 CI/CD — GitHub Actions                                       │  │
│  │  Lint → Test → Build Docker → Push to Registry                   │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- - Docker & Docker Compose
  - - (Optional) MLflow tracking server
   
    - ### Local Development
   
    - ```bash
      # Clone the repository
      git clone https://github.com/BrianKeith2027/mlops-clinical-trial-predictor.git
      cd mlops-clinical-trial-predictor

      # Create virtual environment
      python -m venv venv
      source venv/bin/activate  # Windows: venv\Scripts\activate

      # Install dependencies
      pip install -r requirements.txt

      # Train the model (logs to MLflow)
      python src/model/train.py

      # Start the API server
      uvicorn src.api.main:app --reload --port 8000

      # Open the Streamlit demo
      streamlit run src/app/demo.py
      ```

      ### Docker Deployment

      ```bash
      # Build and run all services
      docker-compose up --build

      # Services:
      #   API:      http://localhost:8000
      #   MLflow:   http://localhost:5000
      #   Streamlit: http://localhost:8501
      #   API Docs: http://localhost:8000/docs
      ```

      ## 🔌 API Reference

      ### POST `/predict`

      Predict clinical trial success probability.

      ```bash
      curl -X POST http://localhost:8000/predict \
        -H "Content-Type: application/json" \
        -d '{
          "phase": "Phase III",
          "therapeutic_area": "Oncology",
          "sponsor_type": "Industry",
          "enrollment": 450,
          "trial_design": "Randomized Controlled",
          "endpoint_type": "Overall Survival",
          "num_sites": 85,
          "duration_months": 36,
          "has_biomarker": true,
          "prior_phase_success": true
        }'
      ```

      **Response:**

      ```json
      {
        "trial_id": "pred_20260306_143022",
        "success_probability": 0.72,
        "risk_level": "Medium",
        "confidence_interval": [0.65, 0.79],
        "top_risk_factors": [
          {"feature": "therapeutic_area", "impact": -0.15, "direction": "negative"},
          {"feature": "enrollment", "impact": 0.12, "direction": "positive"},
          {"feature": "has_biomarker", "impact": 0.09, "direction": "positive"}
        ],
        "model_version": "1.2.0",
        "timestamp": "2026-03-06T14:30:22Z"
      }
      ```

      ### GET `/health`

      ```json
      {"status": "healthy", "model_loaded": true, "version": "1.2.0"}
      ```

      ### GET `/model-info`

      ```json
      {
        "model_type": "XGBoost",
        "training_date": "2026-03-01",
        "auc_roc": 0.847,
        "features": 10,
        "training_samples": 12500
      }
      ```

      ## 🧠 Model Details

      ### Training Pipeline

      The model is an **XGBoost classifier** trained on synthetic clinical trial metadata inspired by ClinicalTrials.gov structure. The pipeline includes:

      1. **Data Ingestion** — Load and validate trial records
      2. 2. **Feature Engineering** — Encode categorical variables, scale numerics, create interaction features
         3. 3. **Hyperparameter Optimization** — Optuna-based Bayesian search over XGBoost parameters
            4. 4. **Training & Evaluation** — Stratified k-fold cross-validation with AUC-ROC primary metric
               5. 5. **Explainability** — SHAP values computed for global and local feature importance
                  6. 6. **Model Registration** — Best model logged to MLflow with signature and input example
                    
                     7. ### Performance
                    
                     8. | Metric | Score |
                     9. |--------|-------|
                     10. | AUC-ROC | 0.847 |
                     11. | Precision | 0.81 |
                     12. | Recall | 0.76 |
                     13. | F1-Score | 0.78 |
                     14. | Log Loss | 0.42 |
                    
                     15. ### Feature Importance (SHAP)
                    
                     16. | Rank | Feature | Mean |SHAP| |
                     17. |------|---------|-----------------|
                     18. | 1 | `phase` | 0.182 |
                     19. | 2 | `therapeutic_area` | 0.156 |
                     20. | 3 | `prior_phase_success` | 0.134 |
                     21. | 4 | `enrollment` | 0.098 |
                     22. | 5 | `has_biomarker` | 0.087 |
                    
                     23. ## 📁 Project Structure
                    
                     24. ```
                         mlops-clinical-trial-predictor/
                         │
                         ├── README.md
                         ├── requirements.txt
                         ├── Dockerfile
                         ├── docker-compose.yml
                         ├── .gitignore
                         ├── LICENSE
                         │
                         ├── .github/
                         │   └── workflows/
                         │       └── ci.yml                  # GitHub Actions CI/CD pipeline
                         │
                         ├── config/
                         │   ├── model_config.yaml           # Model hyperparameters & settings
                         │   └── logging_config.yaml         # Logging configuration
                         │
                         ├── src/
                         │   ├── __init__.py
                         │   │
                         │   ├── data/
                         │   │   ├── __init__.py
                         │   │   ├── generate_synthetic.py   # Synthetic trial data generator
                         │   │   ├── preprocess.py           # Feature engineering pipeline
                         │   │   └── validate.py             # Great Expectations data checks
                         │   │
                         │   ├── model/
                         │   │   ├── __init__.py
                         │   │   ├── train.py                # Training pipeline with MLflow
                         │   │   ├── predict.py              # Inference logic
                         │   │   ├── explain.py              # SHAP explainability module
                         │   │   └── registry.py             # MLflow model registry helpers
                         │   │
                         │   ├── api/
                         │   │   ├── __init__.py
                         │   │   ├── main.py                 # FastAPI application
                         │   │   ├── schemas.py              # Pydantic request/response models
                         │   │   └── middleware.py           # Logging & error handling
                         │   │
                         │   └── app/
                         │       └── demo.py                 # Streamlit demo interface
                         │
                         ├── tests/
                         │   ├── __init__.py
                         │   ├── test_model.py               # Model training & prediction tests
                         │   ├── test_api.py                 # API endpoint tests
                         │   ├── test_data.py                # Data validation tests
                         │   └── conftest.py                 # Shared fixtures
                         │
                         ├── notebooks/
                         │   ├── 01_eda.ipynb                # Exploratory data analysis
                         │   └── 02_model_experiments.ipynb  # Model comparison experiments
                         │
                         ├── data/
                         │   └── sample/
                         │       └── clinical_trials.csv     # Sample training data
                         │
                         └── models/
                             └── .gitkeep                    # Trained model artifacts directory
                         ```

                         ## 🛠️ Tech Stack

                         | Category | Technology |
                         |----------|-----------|
                         | **ML Framework** | XGBoost, scikit-learn, Optuna |
                         | **Explainability** | SHAP |
                         | **API Framework** | FastAPI, Uvicorn, Pydantic |
                         | **Experiment Tracking** | MLflow |
                         | **Containerization** | Docker, Docker Compose |
                         | **CI/CD** | GitHub Actions |
                         | **Dashboard** | Streamlit |
                         | **Data Validation** | Great Expectations, Pandera |
                         | **Testing** | pytest, httpx (async API tests) |
                         | **Language** | Python 3.10+ |

                         ## 🐳 Docker Services

                         ```yaml
                         services:
                           api:         # FastAPI prediction service → port 8000
                           mlflow:      # MLflow tracking UI        → port 5000
                           streamlit:   # Interactive demo          → port 8501
                         ```

                         ## 🔄 CI/CD Pipeline (GitHub Actions)

                         The `.github/workflows/ci.yml` pipeline runs on every push and PR:

                         1. **Lint** — `ruff` for style, `mypy` for type checking
                         2. 2. **Test** — `pytest` with coverage report
                            3. 3. **Build** — Docker image build and validation
                               4. 4. **Security** — `bandit` for security scanning
                                 
                                  5. ## 📈 End-Use Scenarios
                                 
                                  6. | Scenario | Who Uses It | How |
                                  7. |----------|-------------|-----|
                                  8. | **Trial Risk Scoring** | Clinical Ops Analysts | POST trial params to `/predict`, get success probability |
                                  9. | **Portfolio Prioritization** | VP of R&D | Batch-score all active trials, rank by predicted success |
                                  10. | **Protocol Optimization** | Biostatisticians | Adjust trial parameters in Streamlit, see real-time impact |
                                  11. | **Regulatory Strategy** | Regulatory Affairs | Identify risk factors via SHAP to strengthen submissions |
                                  12. | **Investment Diligence** | BD&L Teams | Score acquisition targets' pipeline probability |
                                 
                                  13. ## 🔮 Future Improvements
                                 
                                  14. - Add Kubernetes deployment manifests (Helm chart)
                                      - - Implement A/B model serving for champion/challenger testing
                                        - - Add Evidently AI for production data drift monitoring
                                          - - Integrate real ClinicalTrials.gov API data via ETL pipeline
                                            - - Build Slack/Teams alerting for model performance degradation
                                             
                                              - ## 👤 Author
                                             
                                              - **Brian Stratton**
                                              - Senior Data Engineer | AI/ML Engineer | Doctoral Researcher
                                             
                                              - [LinkedIn](https://www.linkedin.com/in/briankstratton/) | [GitHub](https://github.com/BrianKeith2027)
                                             
                                              - ## 📄 License
                                             
                                              - This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
