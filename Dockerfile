# ============================================================
# MLOps Clinical Trial Predictor — Dockerfile
# Author: Brian Stratton
# Multi-stage build for production FastAPI serving
# ============================================================

FROM python:3.11-slim AS base

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
        curl \
            && rm -rf /var/lib/apt/lists/*

            # Python dependencies
            COPY requirements.txt .
            RUN pip install --no-cache-dir -r requirements.txt

            # ── Production Stage ────────────────────────────────────────
            FROM base AS production

            COPY src/ ./src/
            COPY config/ ./config/
            COPY models/ ./models/

            ENV PYTHONPATH=/app
            ENV MODEL_PATH=/app/models/xgb_trial_predictor.joblib

            EXPOSE 8000

            HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
                CMD curl -f http://localhost:8000/health || exit 1

                CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
