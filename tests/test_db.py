from collections.abc import Iterator
from datetime import timedelta

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import inspect, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from oc_p05_attrition_ml_deployment import db
from oc_p05_attrition_ml_deployment.api import app
from oc_p05_attrition_ml_deployment.db import get_engine
from oc_p05_attrition_ml_deployment.models import ModelInput, ModelOutput
from oc_p05_attrition_ml_deployment.schemas import EmployeeInput

# The documented example of each field is a valid hand-made value.
EMPLOYEE = {name: field.examples[0] for name, field in EmployeeInput.model_fields.items()}


@pytest.fixture
def session() -> Iterator[Session]:
    """A session whose changes are rolled back at the end of the test."""
    with Session(get_engine()) as session:
        yield session
        session.rollback()


def test_connection_works():
    with get_engine().connect() as connection:
        assert connection.execute(text("SELECT 1")).scalar() == 1


def test_tables_exist_after_startup():
    with TestClient(app):
        tables = inspect(get_engine())
        assert tables.has_table("model_inputs")
        assert tables.has_table("model_outputs")


def test_model_input_columns_match_the_employee_fields():
    columns = {column.name for column in ModelInput.__table__.columns}

    assert columns - {"id", "created_at"} == set(EmployeeInput.model_fields)


def test_input_and_output_round_trip(session):
    record = ModelInput(**EMPLOYEE)
    record.output = ModelOutput(score=0.3, alert=True, model_version="test-0")
    session.add(record)
    session.flush()
    session.expire_all()

    stored = session.get(ModelInput, record.id)
    assert stored.poste == EMPLOYEE["poste"]
    assert stored.revenu_mensuel == EMPLOYEE["revenu_mensuel"]
    assert stored.created_at.utcoffset() == timedelta(0)
    assert stored.output.input_id == stored.id
    assert stored.output.score == pytest.approx(0.3)
    assert stored.output.alert is True
    assert stored.output.model_version == "test-0"
    assert stored.output.created_at.utcoffset() == timedelta(0)


def test_second_output_for_the_same_input_is_rejected(session):
    record = ModelInput(**EMPLOYEE)
    record.output = ModelOutput(score=0.3, alert=True, model_version="test-0")
    session.add(record)
    session.flush()
    session.add(ModelOutput(input_id=record.id, score=0.4, alert=True, model_version="test-0"))

    with pytest.raises(IntegrityError):
        session.flush()


def test_output_without_input_is_rejected(session):
    session.add(ModelOutput(input_id=999_999, score=0.3, alert=True, model_version="test-0"))

    with pytest.raises(IntegrityError):
        session.flush()


def test_get_engine_fails_without_database_url(monkeypatch):
    get_engine.cache_clear()
    monkeypatch.delenv("DATABASE_URL", raising=False)
    try:
        with pytest.raises(RuntimeError, match="DATABASE_URL"):
            get_engine()
    finally:
        get_engine.cache_clear()


def test_engine_sets_a_connect_timeout(monkeypatch):
    captured = {}
    monkeypatch.setattr(db, "create_engine", lambda url, **kwargs: captured.update(kwargs))
    get_engine.cache_clear()
    try:
        get_engine()
    finally:
        get_engine.cache_clear()

    assert captured["connect_args"] == {"connect_timeout": db.CONNECT_TIMEOUT_SECONDS}
