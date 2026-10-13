# syntax=docker/dockerfile:1

ARG PYTHON_VERSION=3.13

# Stage 1: train the model inside the build so the image is fully reproducible
FROM python:${PYTHON_VERSION}-slim AS builder
ARG MODEL_VERSION=v0.1
ENV PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app/src
WORKDIR /app
COPY requirements.txt requirements-train.txt ./
RUN pip install -r requirements-train.txt
COPY src ./src
RUN python -m triage.train --version "${MODEL_VERSION}" --out /app/artifacts

# Stage 2: slim runtime image with only the serving dependencies and the baked model
FROM python:${PYTHON_VERSION}-slim
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app/src \
    MODEL_DIR=/app/artifacts
WORKDIR /app
COPY requirements.txt ./
RUN pip install -r requirements.txt
COPY src ./src
COPY --from=builder /app/artifacts ./artifacts

EXPOSE 80
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:80/health', timeout=2)"

CMD ["uvicorn", "triage.app:app", "--host", "0.0.0.0", "--port", "80"]
