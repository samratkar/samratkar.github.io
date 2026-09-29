# Lab 4 -> Lab 5 transition: package the FreshMart serving stack as an immutable
# image. What ships is not "the scripts" - it is this image, identified by a
# digest, and Lab 5's blue/green, canary and rollback all act on it.
FROM python:3.11-slim

WORKDIR /app

# dependencies first, so a code change does not invalidate the pip layer
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

# serving code, the feature-store config and the MLflow store are baked in
COPY serving/ ./serving/
COPY feature_repo/ ./feature_repo/
COPY artifacts/ ./artifacts/
COPY src/monitor.py ./src/monitor.py

ENV MLFLOW_TRACKING_URI=sqlite:///artifacts/mlflow.db
ENV MODEL_ALIAS=production
ENV FEAST_REPO=feature_repo
EXPOSE 8000

# Readiness is the container's health signal - the same endpoint Lab 3 built.
# start-period matters: this service takes ~30s to load a model and a feature
# store, and without it the orchestrator kills the container before it finishes.
HEALTHCHECK --interval=10s --timeout=3s --retries=5 --start-period=60s \
  CMD python -c "import requests,sys; sys.exit(0 if requests.get('http://localhost:8000/ready').status_code==200 else 1)"

CMD ["uvicorn", "serving.api:app", "--host", "0.0.0.0", "--port", "8000"]
