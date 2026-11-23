import hashlib
import json
import logging
import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import joblib
import pandas as pd
from fastapi import Depends, FastAPI, Request
from sklearn.pipeline import Pipeline
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from oc_p05_attrition_ml_deployment.db import create_tables, get_engine, get_session
from oc_p05_attrition_ml_deployment.features import build_inputs
from oc_p05_attrition_ml_deployment.models import PredictionLog
from oc_p05_attrition_ml_deployment.schemas import EmployeeInput, PredictionOutput

DEFAULT_MODEL_DIR = "models/"
PIPELINE_FILE = "attrition_pipeline.joblib"
METADATA_FILE = "attrition_metadata.json"

logger = logging.getLogger(__name__)


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


def log_prediction(session: Session, employee: EmployeeInput, output: PredictionOutput) -> None:
    """Store one prediction; a failed write is logged and never breaks the response."""
    try:
        session.add(
            PredictionLog(
                model_version=output.model_version,
                inputs=employee.model_dump(mode="json"),
                score=output.score,
                alert=output.alert,
            )
        )
        session.commit()
    except SQLAlchemyError:
        # No rollback here: get_session closes the session, which ends the transaction.
        logger.exception("Could not store the prediction log")


@app.post("/predict")
def predict(
    employee: EmployeeInput,
    request: Request,
    session: Annotated[Session, Depends(get_session)],
) -> PredictionOutput:
    """Predict the attrition risk of one employee.

    Takes the raw employee fields and returns a model score between 0 and 1, an alert flag
    (true when the score is at or above the alert threshold stored with the model)
    and the version of the model that answered. Invalid fields give a 422 response.
    Each successful prediction is stored in the database; if the write fails, the
    prediction is still returned and the error is logged.
    """
    model: LoadedModel = request.app.state.model
    inputs = build_inputs(pd.DataFrame([employee.model_dump()]))
    probability = float(model.pipeline.predict_proba(inputs)[0, 1])
    output = PredictionOutput(
        score=probability,
        alert=probability >= model.threshold,
        model_version=model.version,
    )
    log_prediction(session, employee, output)
    return output
