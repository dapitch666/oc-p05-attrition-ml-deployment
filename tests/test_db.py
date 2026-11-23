from collections.abc import Iterator
from datetime import timedelta

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import inspect, text
from sqlalchemy.orm import Session

from oc_p05_attrition_ml_deployment.api import app
from oc_p05_attrition_ml_deployment.db import get_engine
from oc_p05_attrition_ml_deployment.models import PredictionLog


@pytest.fixture
def session() -> Iterator[Session]:
    """A session whose changes are rolled back at the end of the test."""
    with Session(get_engine()) as session:
        yield session
        session.rollback()


def test_connection_works():
    with get_engine().connect() as connection:
        assert connection.execute(text("SELECT 1")).scalar() == 1


def test_table_exists_after_startup():
    with TestClient(app):
        assert inspect(get_engine()).has_table("prediction_logs")


def test_prediction_log_round_trip(session):
    log = PredictionLog(
        model_version="test-0",
        inputs={"age": 35, "poste": "Consultant"},
        score=0.3,
        alert=True,
    )
    session.add(log)
    session.flush()
    session.expire(log)

    stored = session.get(PredictionLog, log.id)
    assert stored.model_version == "test-0"
    assert stored.inputs == {"age": 35, "poste": "Consultant"}
    assert stored.score == pytest.approx(0.3)
    assert stored.alert is True
    assert stored.created_at.utcoffset() == timedelta(0)


def test_get_engine_fails_without_database_url(monkeypatch):
    get_engine.cache_clear()
    monkeypatch.delenv("DATABASE_URL", raising=False)
    try:
        with pytest.raises(RuntimeError, match="DATABASE_URL"):
            get_engine()
    finally:
        get_engine.cache_clear()
