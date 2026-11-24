import json
import re
from collections.abc import Iterator
from dataclasses import replace
from importlib.metadata import version
from pathlib import Path
from typing import get_args
from unittest.mock import Mock

import joblib
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import insert, select
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from oc_p05_attrition_ml_deployment.api import app
from oc_p05_attrition_ml_deployment.db import get_engine, get_session
from oc_p05_attrition_ml_deployment.models import ModelInput, ModelOutput
from oc_p05_attrition_ml_deployment.schemas import EmployeeInput

MODEL_DIR = Path(__file__).parent.parent / "models"

VALID_EMPLOYEE = {
    "age": 35,
    "revenu_mensuel": 5000,
    "nombre_experiences_precedentes": 2,
    "annees_experience_totale": 10,
    "annees_dans_l_entreprise": 5,
    "annees_dans_le_poste_actuel": 2,
    "annees_depuis_la_derniere_promotion": 1,
    "annees_sous_responsable_actuel": 2,
    "satisfaction_employee_environnement": 3,
    "satisfaction_employee_nature_travail": 3,
    "satisfaction_employee_equipe": 3,
    "satisfaction_employee_equilibre_pro_perso": 3,
    "note_evaluation_precedente": 3,
    "niveau_education": 3,
    "augmentation_salaire_precedente": 12,
    "nombre_participation_pee": 1,
    "nb_formations_suivies": 2,
    "distance_domicile_travail": 10,
    "frequence_deplacement": "Occasionnel",
    "genre": "F",
    "heures_supplementaires": "Non",
    "statut_marital": "Marié(e)",
    "poste": "Consultant",
    "domaine_etude": "Infra & Cloud",
}


@pytest.fixture
def client() -> Iterator[TestClient]:
    with TestClient(app) as client:
        yield client


def test_api_version_comes_from_package_metadata(client):
    expected = version("oc-p05-attrition-ml-deployment")
    assert app.version == expected
    assert client.get("/openapi.json").json()["info"]["version"] == expected


def test_health_returns_ok(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_returns_score_alert_and_version(client):
    response = client.post("/predict", json=VALID_EMPLOYEE)

    assert response.status_code == 200
    body = response.json()
    threshold = json.loads((MODEL_DIR / "attrition_metadata.json").read_text())["threshold"]
    assert 0 <= body["score"] <= 1
    assert body["alert"] == (body["score"] >= threshold)
    assert re.fullmatch(r"1\.0\.0-[0-9a-f]{8}", body["model_version"])


def test_predict_alert_follows_threshold(client):
    score = client.post("/predict", json=VALID_EMPLOYEE).json()["score"]
    loaded = app.state.model

    for threshold, expected in ((score, True), (score + 0.01, False)):
        app.state.model = replace(loaded, threshold=threshold)
        assert client.post("/predict", json=VALID_EMPLOYEE).json()["alert"] is expected


@pytest.mark.parametrize(
    "overrides",
    [
        {"age": "thirty"},
        {"age": 17},
        {"revenu_mensuel": 100_001},
        {"satisfaction_employee_equipe": 5},
        {"augmentation_salaire_precedente": 101},
        {"poste": "Astronaute"},
        {"genre": "X"},
        {"unknown_field": 1},
    ],
)
def test_predict_rejects_invalid_value(client, overrides):
    response = client.post("/predict", json={**VALID_EMPLOYEE, **overrides})

    assert response.status_code == 422


@pytest.mark.parametrize("field", list(VALID_EMPLOYEE))
def test_predict_rejects_missing_field(client, field):
    payload = {key: value for key, value in VALID_EMPLOYEE.items() if key != field}

    assert client.post("/predict", json=payload).status_code == 422


@pytest.mark.parametrize(
    "overrides",
    [
        {"annees_experience_totale": 4, "annees_dans_l_entreprise": 5},
        {"annees_dans_le_poste_actuel": 6},
        {"annees_depuis_la_derniere_promotion": 6},
        {"annees_sous_responsable_actuel": 6},
    ],
)
def test_predict_rejects_incoherent_years(client, overrides):
    response = client.post("/predict", json={**VALID_EMPLOYEE, **overrides})

    assert response.status_code == 422


def test_years_equal_to_tenure_are_accepted(client):
    payload = {
        **VALID_EMPLOYEE,
        "annees_experience_totale": 5,
        "annees_dans_l_entreprise": 5,
        "annees_dans_le_poste_actuel": 5,
        "annees_depuis_la_derniere_promotion": 5,
        "annees_sous_responsable_actuel": 5,
    }

    assert client.post("/predict", json=payload).status_code == 200


def test_literals_match_pipeline_encoder_categories():
    preprocessing = joblib.load(MODEL_DIR / "attrition_pipeline.joblib").named_steps[
        "preprocessing"
    ]
    for name, encoder, columns in preprocessing.transformers_:
        if name == "numeric":
            continue
        for column, categories in zip(columns, encoder.categories_, strict=True):
            literal = get_args(EmployeeInput.model_fields[column].annotation)
            assert set(literal) == set(categories), column


def test_startup_fails_when_model_dir_is_missing(monkeypatch, tmp_path):
    monkeypatch.setenv("MODEL_DIR", str(tmp_path / "missing"))

    with pytest.raises(FileNotFoundError, match="MODEL_DIR"):
        with TestClient(app):
            pass


def test_revenu_mensuel_at_the_upper_bound_is_accepted(client):
    response = client.post("/predict", json={**VALID_EMPLOYEE, "revenu_mensuel": 100_000})

    assert response.status_code == 200


def stored_inputs() -> list[ModelInput]:
    with Session(get_engine()) as session:
        return list(session.scalars(select(ModelInput)))


def stored_outputs() -> list[ModelOutput]:
    with Session(get_engine()) as session:
        return list(session.scalars(select(ModelOutput)))


def break_database(monkeypatch, *, on_flush=False, on_commit=None):
    """Make the API sessions fail on flush, or on the n-th commit."""

    class BrokenSession(Session):
        commits = 0

        def flush(self, objects=None):
            if on_flush:
                raise OperationalError("INSERT", {}, Exception("database down"))
            super().flush(objects)

        def commit(self):
            self.commits += 1
            if self.commits == on_commit:
                raise OperationalError("INSERT", {}, Exception("database down"))
            super().commit()

    def broken_session():
        with BrokenSession(get_engine()) as session:
            yield session

    monkeypatch.setitem(app.dependency_overrides, get_session, broken_session)


def test_predict_stores_the_input_and_its_linked_output(client):
    body = client.post("/predict", json=VALID_EMPLOYEE).json()

    [stored_input] = stored_inputs()
    [stored_output] = stored_outputs()
    for field, value in VALID_EMPLOYEE.items():
        assert getattr(stored_input, field) == value
    assert stored_output.input_id == stored_input.id
    assert stored_output.score == pytest.approx(body["score"])
    assert stored_output.alert == body["alert"]
    assert stored_output.model_version == body["model_version"]


def test_database_failure_on_the_input_gives_503_and_skips_the_model(client, monkeypatch):
    pipeline = Mock()
    monkeypatch.setattr(app.state, "model", replace(app.state.model, pipeline=pipeline))
    break_database(monkeypatch, on_flush=True)

    response = client.post("/predict", json=VALID_EMPLOYEE)

    assert response.status_code == 503
    assert "score" not in response.json()
    pipeline.predict_proba.assert_not_called()
    assert stored_inputs() == []
    assert stored_outputs() == []


def test_database_failure_on_the_output_gives_503_and_keeps_the_input(client, monkeypatch):
    break_database(monkeypatch, on_commit=2)

    response = client.post("/predict", json=VALID_EMPLOYEE)

    assert response.status_code == 503
    assert "score" not in response.json()
    assert len(stored_inputs()) == 1
    assert stored_outputs() == []


def test_model_failure_keeps_the_input_without_output(monkeypatch):
    pipeline = Mock()
    pipeline.predict_proba.side_effect = RuntimeError("model failure")
    with TestClient(app, raise_server_exceptions=False) as client:
        monkeypatch.setattr(app.state, "model", replace(app.state.model, pipeline=pipeline))
        response = client.post("/predict", json=VALID_EMPLOYEE)

    assert response.status_code == 500
    assert len(stored_inputs()) == 1
    assert stored_outputs() == []


def test_invalid_request_stores_nothing(client):
    response = client.post("/predict", json={**VALID_EMPLOYEE, "age": 17})

    assert response.status_code == 422
    assert stored_inputs() == []
    assert stored_outputs() == []


def test_failed_write_does_not_log_employee_values(client, monkeypatch, caplog):
    payload = {
        **VALID_EMPLOYEE,
        "revenu_mensuel": 4321,
        "statut_marital": "Divorcé(e)",
        "poste": "Tech Lead",
    }

    class FailingInsertSession(Session):
        def flush(self, objects=None):
            # A real database error (smallint overflow): its message lists the statement parameters.
            self.connection().execute(
                insert(ModelInput).values(
                    age=70_000,
                    revenu_mensuel=payload["revenu_mensuel"],
                    statut_marital=payload["statut_marital"],
                    poste=payload["poste"],
                )
            )

    def failing_session():
        with FailingInsertSession(get_engine()) as session:
            yield session

    monkeypatch.setitem(app.dependency_overrides, get_session, failing_session)
    with caplog.at_level("ERROR"):
        response = client.post("/predict", json=payload)

    assert response.status_code == 503
    assert "Could not store the model input" in caplog.text
    for value in ("4321", "Divorcé(e)", "Tech Lead"):
        assert value not in caplog.text
