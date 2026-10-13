"""FastAPI service exposing GET /health and POST /predict."""

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import joblib
import numpy as np
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from triage.features import FEATURES

DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[2] / "artifacts"

Feature = Annotated[float, Field(allow_inf_nan=False)]


class PatientFeatures(BaseModel):
    """The ten standardized diabetes features of one patient check-in."""

    age: Feature
    sex: Feature
    bmi: Feature
    bp: Feature
    s1: Feature
    s2: Feature
    s3: Feature
    s4: Feature
    s5: Feature
    s6: Feature


class Prediction(BaseModel):
    prediction: float


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_dir = Path(os.environ.get("MODEL_DIR", DEFAULT_MODEL_DIR))
    app.state.model = joblib.load(model_dir / "model.joblib")
    metrics = json.loads((model_dir / "metrics.json").read_text())
    app.state.model_version = metrics["model_version"]
    yield


app = FastAPI(title="Diabetes clinic triage", lifespan=lifespan)


def _field_name(error: dict) -> str:
    parts = [str(p) for p in error["loc"] if p != "body" and not isinstance(p, int)]
    return ".".join(parts) or "body"


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    details = [
        {"field": _field_name(err), "issue": err["msg"], "type": err["type"]}
        for err in exc.errors()
    ]
    return JSONResponse(
        status_code=422,
        content={
            "error": "validation_error",
            "message": "Invalid request payload",
            "details": details,
        },
    )


@app.exception_handler(Exception)
async def unexpected_error_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"error": "internal_error", "message": "Unexpected server error"},
    )


@app.get("/health")
def health(request: Request):
    return {"status": "ok", "model_version": request.app.state.model_version}


@app.post("/predict", response_model=Prediction)
def predict(patient: PatientFeatures, request: Request):
    row = np.array([[getattr(patient, name) for name in FEATURES]])
    value = float(request.app.state.model.predict(row)[0])
    return {"prediction": value}
