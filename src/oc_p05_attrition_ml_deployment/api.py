import hashlib
import json
import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, Request
from sklearn.pipeline import Pipeline

from oc_p05_attrition_ml_deployment import models  # noqa: F401  # registers the tables
from oc_p05_attrition_ml_deployment.db import create_tables, get_engine
from oc_p05_attrition_ml_deployment.features import build_inputs
from oc_p05_attrition_ml_deployment.schemas import EmployeeInput, PredictionOutput

DEFAULT_MODEL_DIR = "models/"
PIPELINE_FILE = "attrition_pipeline.joblib"
METADATA_FILE = "attrition_metadata.json"


@dataclass(frozen=True)
class LoadedModel:
    """The pipeline with what the endpoints need to answer."""

    pipeline: Pipeline
    threshold: float
    version: str


def load_model(model_dir: Path) -> LoadedModel:
    """Load the pipeline and its metadata, and compute the model version."""
    pipeline_path = model_dir / PIPELINE_FILE
    metadata_path = model_dir / METADATA_FILE
    for path in (pipeline_path, metadata_path):
        if not path.is_file():
            raise FileNotFoundError(f"Model file not found: {path}. Check MODEL_DIR.")
    metadata = json.loads(metadata_path.read_text())
    digest = hashlib.sha256(pipeline_path.read_bytes()).hexdigest()[:8]
    return LoadedModel(
        pipeline=joblib.load(pipeline_path),
        threshold=metadata["threshold"],
        version=f"{metadata['model_version']}-{digest}",
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    """Load the model and create the tables once, before the first request."""
    app.state.model = load_model(Path(os.environ.get("MODEL_DIR", DEFAULT_MODEL_DIR)))
    create_tables(get_engine())
    yield


app = FastAPI(
    title="Futurisys attrition API",
    description="API serving the employee attrition classifier.",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict[str, str]:
    """Report that the API is running."""
    return {"status": "ok"}


@app.post("/predict")
def predict(employee: EmployeeInput, request: Request) -> PredictionOutput:
    """Predict the attrition risk of one employee.

    Takes the raw employee fields and returns a model score between 0 and 1, an alert flag
    (true when the score is at or above the alert threshold stored with the model)
    and the version of the model that answered. Invalid fields give a 422 response.
    """
    model: LoadedModel = request.app.state.model
    inputs = build_inputs(pd.DataFrame([employee.model_dump()]))
    probability = float(model.pipeline.predict_proba(inputs)[0, 1])
    return PredictionOutput(
        score=probability,
        alert=probability >= model.threshold,
        model_version=model.version,
    )
