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
from sqlalchemy import insert, select, text
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from oc_p05_attrition_ml_deployment import api
from oc_p05_attrition_ml_deployment.api import app
from oc_p05_attrition_ml_deployment.auth import create_api_key, hash_key, revoke_api_key
from oc_p05_attrition_ml_deployment.db import get_admin_engine, get_engine, get_session
from oc_p05_attrition_ml_deployment.models import ApiError, ApiKey, ModelInput, ModelOutput
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
def client(api_key: str) -> Iterator[TestClient]:
    with TestClient(app, headers={"X-API-Key": api_key}) as client:
        yield client


def test_api_version_comes_from_package_metadata(client):
    expected = version("oc-p05-attrition-ml-deployment")
    assert app.version == expected
    assert client.get("/openapi.json").json()["info"]["version"] == expected


@pytest.mark.parametrize("environment", ["dev", "test", "prod"])
def test_health_returns_status_version_and_environment(monkeypatch, environment):
    monkeypatch.setenv("APP_ENV", environment)

    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "version": version("oc-p05-attrition-ml-deployment"),
        "environment": environment,
    }


def test_startup_fails_when_app_env_is_missing(monkeypatch):
    monkeypatch.delenv("APP_ENV")

    with pytest.raises(RuntimeError, match="APP_ENV must be one of: dev, test, prod"):
        with TestClient(app):
            pass


@pytest.mark.parametrize("value", ["", "staging", "PROD"])
def test_startup_fails_when_app_env_is_unknown(monkeypatch, value):
    monkeypatch.setenv("APP_ENV", value)

    with pytest.raises(RuntimeError, match="APP_ENV must be one of: dev, test, prod"):
        with TestClient(app):
            pass


def test_health_fields_are_documented(client):
    schemas = client.get("/openapi.json").json()["components"]["schemas"]

    properties = schemas["HealthOutput"]["properties"]
    assert properties["status"]["description"]
    assert properties["version"]["description"]
    assert properties["environment"]["description"]
    assert properties["environment"]["enum"] == ["dev", "test", "prod"]


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
            if on_flush and self.new:
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


def test_model_failure_keeps_the_input_without_output(monkeypatch, api_key):
    pipeline = Mock()
    pipeline.predict_proba.side_effect = RuntimeError("model failure")
    with TestClient(app, raise_server_exceptions=False, headers={"X-API-Key": api_key}) as client:
        monkeypatch.setattr(app.state, "model", replace(app.state.model, pipeline=pipeline))
        response = client.post("/predict", json=VALID_EMPLOYEE)

    assert response.status_code == 500
    assert len(stored_inputs()) == 1
    assert stored_outputs() == []


def test_invalid_request_stores_no_model_row(client):
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
            if not self.new:
                return
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


def anonymous_client() -> TestClient:
    return TestClient(app, raise_server_exceptions=False)


def test_health_is_open_without_a_key():
    with anonymous_client() as client:
        assert client.get("/health").status_code == 200


def test_predict_without_a_key_gives_401_and_stores_nothing():
    with anonymous_client() as client:
        response = client.post("/predict", json=VALID_EMPLOYEE)

    assert response.status_code == 401
    assert response.json() == {"detail": "Invalid or missing API key."}
    assert stored_inputs() == []


def test_predict_with_an_unknown_key_gives_the_same_401():
    with anonymous_client() as client:
        missing = client.post("/predict", json=VALID_EMPLOYEE)
        unknown = client.post("/predict", json=VALID_EMPLOYEE, headers={"X-API-Key": "nope"})

    assert unknown.status_code == 401
    assert unknown.json() == missing.json()
    assert stored_inputs() == []


def test_predict_with_a_revoked_key_gives_the_same_401(client):
    with Session(get_admin_engine()) as session:
        key_id = session.scalars(select(ApiKey.id)).one()
        revoke_api_key(session, key_id)

    response = client.post("/predict", json=VALID_EMPLOYEE)

    assert response.status_code == 401
    assert response.json() == {"detail": "Invalid or missing API key."}
    assert stored_inputs() == []


def test_401_carries_the_www_authenticate_header():
    with anonymous_client() as client:
        response = client.post("/predict", json=VALID_EMPLOYEE)

    assert response.headers["WWW-Authenticate"] == "APIKey"


def test_a_bad_key_is_rejected_before_the_body_is_validated():
    with anonymous_client() as client:
        response = client.post("/predict", json={**VALID_EMPLOYEE, "age": 17})

    assert response.status_code == 401


def test_predict_links_the_input_to_the_key_that_made_the_call(client, api_key):
    with Session(get_admin_engine()) as session:
        other_id, _ = create_api_key(session, "Other client")
        key_id = session.scalars(
            select(ApiKey.id).where(ApiKey.key_hash == hash_key(api_key))
        ).one()

    client.post("/predict", json=VALID_EMPLOYEE)

    [stored_input] = stored_inputs()
    assert stored_input.api_key_id == key_id
    assert stored_input.api_key_id != other_id


def test_only_the_hash_of_the_key_is_stored(api_key):
    with get_engine().connect() as connection:
        rows = connection.execute(text("SELECT * FROM api_keys")).fetchall()

    assert [hash_key(api_key) in row for row in rows] == [True]
    assert all(api_key not in str(value) for row in rows for value in row)


def test_database_failure_while_checking_the_key_gives_503(api_key, monkeypatch, caplog):
    class BrokenLookupSession(Session):
        def scalar(self, *args, **kwargs):
            raise OperationalError("SELECT", {}, Exception("database down"))

    def broken_session():
        with BrokenLookupSession(get_engine()) as session:
            yield session

    pipeline = Mock()
    monkeypatch.setattr(app.state, "model", replace(app.state.model, pipeline=pipeline))
    monkeypatch.setitem(app.dependency_overrides, get_session, broken_session)
    with caplog.at_level("ERROR"), TestClient(app, headers={"X-API-Key": api_key}) as client:
        response = client.post("/predict", json=VALID_EMPLOYEE)

    assert response.status_code == 503
    pipeline.predict_proba.assert_not_called()
    assert stored_inputs() == []
    for secret in (api_key, hash_key(api_key)):
        assert secret not in response.text
        assert secret not in caplog.text


def test_swagger_declares_the_key_scheme_and_the_401_header():
    schema = app.openapi()

    scheme = next(iter(schema["components"]["securitySchemes"].values()))
    assert scheme["type"] == "apiKey"
    assert scheme["name"] == "X-API-Key"
    assert scheme["in"] == "header"
    responses = schema["paths"]["/predict"]["post"]["responses"]
    assert "WWW-Authenticate" in responses["401"]["headers"]
    assert "503" in responses


def stored_errors() -> list[ApiError]:
    with Session(get_engine()) as session:
        return list(session.scalars(select(ApiError)))


def stored_error_rows() -> str:
    """The whole rows as PostgreSQL holds them, to search them for forbidden content."""
    with get_engine().connect() as connection:
        rows = connection.execute(text("SELECT row_to_json(api_errors)::text FROM api_errors"))
        return "\n".join(rows.scalars())


def assert_one_401_row(reason: str, key: str | None) -> None:
    [error] = stored_errors()
    assert (error.method, error.path, error.status_code) == ("POST", "/predict", 401)
    assert error.api_key_id is None
    assert error.detail == {"reason": reason}
    assert error.created_at is not None
    if key:
        row = stored_error_rows()
        assert key not in row
        assert hash_key(key) not in row


def test_401_without_a_key_is_logged():
    with anonymous_client() as client:
        client.post("/predict", json=VALID_EMPLOYEE)

    assert_one_401_row("missing", None)


def test_401_with_an_unknown_key_is_logged_without_the_key():
    key = "distinctive-unknown-key"
    with anonymous_client() as client:
        client.post("/predict", json=VALID_EMPLOYEE, headers={"X-API-Key": key})

    assert_one_401_row("invalid", key)


def test_401_with_a_revoked_key_is_logged_without_the_key(client, api_key):
    with Session(get_admin_engine()) as session:
        revoke_api_key(session, session.scalars(select(ApiKey.id)).one())

    client.post("/predict", json=VALID_EMPLOYEE)

    assert_one_401_row("invalid", api_key)


def test_422_is_logged_with_the_key_and_without_the_submitted_values(client):
    payload = {**VALID_EMPLOYEE, "age": 7777, "poste": "Distinctive job"}

    response = client.post("/predict", json=payload)

    assert response.status_code == 422
    [error] = stored_errors()
    assert (error.method, error.path, error.status_code) == ("POST", "/predict", 422)
    with Session(get_engine()) as session:
        assert error.api_key_id == session.scalars(select(ApiKey.id)).one()
    assert [item["loc"] for item in error.detail] == [["body", "age"], ["body", "poste"]]
    assert all(set(item) == {"loc", "type", "msg"} for item in error.detail)
    row = stored_error_rows()
    assert "7777" not in json.dumps(error.detail)
    assert "Distinctive job" not in row


def test_422_response_is_the_default_one(client):
    response = client.post("/predict", json={**VALID_EMPLOYEE, "age": 17})

    assert response.json() == {
        "detail": [
            {
                "type": "greater_than_equal",
                "loc": ["body", "age"],
                "msg": "Input should be greater than or equal to 18",
                "input": 17,
                "ctx": {"ge": 18},
            }
        ]
    }


def test_name_of_an_unknown_field_is_not_stored(client):
    response = client.post("/predict", json={**VALID_EMPLOYEE, "Distinctive name": 1})

    assert response.json()["detail"][0]["loc"] == ["body", "Distinctive name"]
    [error] = stored_errors()
    assert [item["loc"] for item in error.detail] == [["body", "<extra>"]]
    assert "Distinctive name" not in stored_error_rows()


def test_failed_years_check_is_logged_without_the_values(client):
    payload = {**VALID_EMPLOYEE, "annees_experience_totale": 41, "annees_dans_l_entreprise": 43}

    client.post("/predict", json=payload)

    [error] = stored_errors()
    detail = json.dumps(error.detail)
    assert "cannot exceed" in detail
    assert "41" not in detail and "43" not in detail


def test_invalid_body_without_a_key_is_logged_as_a_401_only():
    with anonymous_client() as client:
        response = client.post("/predict", json={**VALID_EMPLOYEE, "age": 17})

    assert response.status_code == 401
    assert_one_401_row("missing", None)


@pytest.mark.parametrize("with_key", [True, False])
def test_malformed_json_is_refused_before_the_key_is_checked(api_key, with_key):
    headers = {"Content-Type": "application/json"}
    if with_key:
        headers["X-API-Key"] = api_key

    with anonymous_client() as client:
        response = client.post("/predict", content='{"poste": "Distinctive', headers=headers)

    assert response.status_code == 422
    [error] = stored_errors()
    assert error.status_code == 422
    assert error.api_key_id is None
    assert [item["type"] for item in error.detail] == ["json_invalid"]
    assert "Distinctive" not in stored_error_rows()


@pytest.mark.parametrize(
    ("headers", "reason"), [({}, "missing"), ({"X-API-Key": "nope"}, "invalid")]
)
def test_failure_to_log_a_401_does_not_change_the_response(monkeypatch, caplog, headers, reason):
    break_database(monkeypatch, on_commit=1)

    with caplog.at_level("ERROR"), anonymous_client() as client:
        response = client.post("/predict", json=VALID_EMPLOYEE, headers=headers)

    assert response.status_code == 401
    assert response.json() == {"detail": "Invalid or missing API key."}
    assert response.headers["WWW-Authenticate"] == "APIKey"
    assert "Could not store the API error" in caplog.text
    assert stored_errors() == []


def test_failure_to_log_a_422_does_not_change_the_response(client, monkeypatch, caplog):
    payload = {**VALID_EMPLOYEE, "age": 17}
    expected = client.post("/predict", json=payload)
    with get_admin_engine().begin() as connection:
        connection.execute(text("TRUNCATE api_errors"))

    class BrokenSession(Session):
        def commit(self):
            raise OperationalError("INSERT", {}, Exception("database down"))

    def broken_session():
        with BrokenSession(get_engine()) as session:
            yield session

    # The handler calls get_session itself, so a dependency override would not reach it.
    monkeypatch.setattr(api, "get_session", broken_session)
    with caplog.at_level("ERROR"):
        response = client.post("/predict", json=payload)

    assert response.status_code == 422
    assert response.json() == expected.json()
    assert "Could not store the API error" in caplog.text
    assert stored_errors() == []


def test_successful_prediction_logs_no_error(client):
    assert client.post("/predict", json=VALID_EMPLOYEE).status_code == 200

    assert stored_errors() == []


def test_503_is_not_logged_as_an_api_error(client, monkeypatch):
    break_database(monkeypatch, on_commit=1)

    assert client.post("/predict", json=VALID_EMPLOYEE).status_code == 503

    assert stored_errors() == []
