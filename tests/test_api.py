import json
import re
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import get_args

import joblib
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from oc_p05_attrition_ml_deployment.api import app
from oc_p05_attrition_ml_deployment.db import get_engine, get_session
from oc_p05_attrition_ml_deployment.models import PredictionLog
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


def stored_logs() -> list[PredictionLog]:
    with Session(get_engine()) as session:
        return list(session.scalars(select(PredictionLog)))


class FailingSession(Session):
    def commit(self) -> None:
        raise OperationalError("INSERT", {}, Exception("database down"))


def test_predict_stores_one_log_with_the_response_values(client):
    body = client.post("/predict", json=VALID_EMPLOYEE).json()

    [log] = stored_logs()
    assert log.inputs == VALID_EMPLOYEE
    assert log.score == pytest.approx(body["score"])
    assert log.alert == body["alert"]
    assert log.model_version == body["model_version"]


def test_predict_still_answers_when_the_database_write_fails(client, monkeypatch, caplog):
    expected = client.post("/predict", json=VALID_EMPLOYEE)

    def failing_session():
        with FailingSession(get_engine()) as session:
            yield session

    monkeypatch.setitem(app.dependency_overrides, get_session, failing_session)
    with caplog.at_level("ERROR"):
        response = client.post("/predict", json=VALID_EMPLOYEE)

    assert response.status_code == 200
    assert response.json() == expected.json()
    assert "Could not store the prediction log" in caplog.text
    assert len(stored_logs()) == 1


def test_invalid_request_stores_nothing(client):
    response = client.post("/predict", json={**VALID_EMPLOYEE, "age": 17})

    assert response.status_code == 422
    assert stored_logs() == []
