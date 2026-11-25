import hashlib
import json
import logging
import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Annotated

import joblib
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Request, Security
from fastapi.security import APIKeyHeader
from sklearn.pipeline import Pipeline
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from oc_p05_attrition_ml_deployment.auth import find_active_key_id
from oc_p05_attrition_ml_deployment.db import get_engine, get_session, missing_tables
from oc_p05_attrition_ml_deployment.features import build_inputs
from oc_p05_attrition_ml_deployment.models import ModelInput, ModelOutput
from oc_p05_attrition_ml_deployment.schemas import EmployeeInput, PredictionOutput

DEFAULT_MODEL_DIR = "models/"
PIPELINE_FILE = "attrition_pipeline.joblib"
METADATA_FILE = "attrition_metadata.json"

DATABASE_UNAVAILABLE = "The database is unavailable, so no prediction can be made."

INVALID_API_KEY = "Invalid or missing API key."

api_key_header = APIKeyHeader(
    name="X-API-Key",
    auto_error=False,
    description="API key issued to the client. It is checked against its stored hash.",
)

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
    """Load the model and check the tables once, before the first request."""
    app.state.model = load_model(Path(os.environ.get("MODEL_DIR", DEFAULT_MODEL_DIR)))
    missing = missing_tables(get_engine())
    if missing:
        raise RuntimeError(
            f"Missing tables: {', '.join(missing)}. "
            "Run the init step: python -m oc_p05_attrition_ml_deployment.cli init"
        )
    yield


app = FastAPI(
    title="Futurisys attrition API",
    description="API serving the employee attrition classifier.",
    version=version("oc-p05-attrition-ml-deployment"),
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict[str, str]:
    """Report that the API is running."""
    return {"status": "ok"}


def authenticate(
    key: Annotated[str | None, Security(api_key_header)],
    session: Annotated[Session, Depends(get_session)],
) -> int:
    """Return the id of the active key sent in X-API-Key; answer 401 or 503 otherwise."""
    # The same answer for a missing, unknown or revoked key, so it reveals nothing about keys.
    unauthorized = HTTPException(
        status_code=401, detail=INVALID_API_KEY, headers={"WWW-Authenticate": "APIKey"}
    )
    if not key:
        raise unauthorized
    try:
        key_id = find_active_key_id(session, key)
    except SQLAlchemyError as error:
        logger.exception("Could not check the API key")
        raise HTTPException(status_code=503, detail=DATABASE_UNAVAILABLE) from error
    if key_id is None:
        raise unauthorized
    return key_id


def store_input(session: Session, employee: EmployeeInput, api_key_id: int) -> int:
    """Commit the raw input with the key that sent it; answer 503 if the database fails."""
    try:
        record = ModelInput(**employee.model_dump(), api_key_id=api_key_id)
        session.add(record)
        # Read the id before the commit, which expires the object.
        session.flush()
        input_id = record.id
        session.commit()
    except SQLAlchemyError as error:
        logger.exception("Could not store the model input")
        raise HTTPException(status_code=503, detail=DATABASE_UNAVAILABLE) from error
    return input_id


def store_output(session: Session, input_id: int, output: PredictionOutput) -> None:
    """Commit the output linked to its input; answer 503 if the database fails."""
    try:
        session.add(
            ModelOutput(
                input_id=input_id,
                score=output.score,
                alert=output.alert,
                model_version=output.model_version,
            )
        )
        session.commit()
    except SQLAlchemyError as error:
        logger.exception("Could not store the model output")
        raise HTTPException(status_code=503, detail=DATABASE_UNAVAILABLE) from error


@app.post(
    "/predict",
    responses={
        401: {
            "description": (
                "The X-API-Key header is missing, unknown or revoked. The response carries a "
                "WWW-Authenticate: APIKey header."
            ),
            "headers": {
                "WWW-Authenticate": {
                    "description": "The authentication scheme expected: APIKey.",
                    "schema": {"type": "string"},
                }
            },
        },
        503: {"description": "The database is unavailable: no prediction is made or returned."},
    },
)
def predict(
    employee: EmployeeInput,
    request: Request,
    session: Annotated[Session, Depends(get_session)],
    api_key_id: Annotated[int, Depends(authenticate)],
) -> PredictionOutput:
    """Predict the attrition risk of one employee.

    Takes the raw employee fields and returns a model score between 0 and 1, an alert flag
    (true when the score is at or above the alert threshold stored with the model)
    and the version of the model that answered. Invalid fields give a 422 response.

    The call needs a valid key in the X-API-Key header, otherwise the response is a 401.
    The key that made the call is stored with the input.

    Every call goes through the database: the input is stored before the model runs and
    the output is stored before the answer is returned. If the database is unavailable,
    the response is a 503 and no prediction is returned. An input whose output could not
    be produced or stored stays in the database without an output.
    """
    model: LoadedModel = request.app.state.model
    input_id = store_input(session, employee, api_key_id)
    inputs = build_inputs(pd.DataFrame([employee.model_dump()]))
    probability = float(model.pipeline.predict_proba(inputs)[0, 1])
    output = PredictionOutput(
        score=probability,
        alert=probability >= model.threshold,
        model_version=model.version,
    )
    store_output(session, input_id, output)
    return output
