import os

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url
from sqlalchemy.exc import OperationalError, ProgrammingError

from oc_p05_attrition_ml_deployment.api import app
from oc_p05_attrition_ml_deployment.cli import main
from oc_p05_attrition_ml_deployment.db import (
    APP_ROLE,
    get_admin_engine,
    get_engine,
    init_database,
    missing_tables,
)
from oc_p05_attrition_ml_deployment.schemas import EmployeeInput

EMPLOYEE = {name: field.examples[0] for name, field in EmployeeInput.model_fields.items()}


@pytest.fixture
def app_password() -> str:
    return os.environ["APP_DB_PASSWORD"]


@pytest.fixture
def drop_all_tables():
    """Drop the tables for a test and rebuild them afterwards."""
    yield
    init_database(get_admin_engine(), os.environ["APP_DB_PASSWORD"])


def insert_input(connection) -> int:
    columns = ", ".join(EMPLOYEE)
    values = ", ".join(f":{name}" for name in EMPLOYEE)
    return connection.execute(
        text(f"INSERT INTO model_inputs ({columns}) VALUES ({values}) RETURNING id"), EMPLOYEE
    ).scalar()


def test_init_creates_the_tables_and_the_role(app_password):
    init_database(get_admin_engine(), app_password)

    assert missing_tables(get_admin_engine()) == []
    with get_admin_engine().connect() as connection:
        found = connection.execute(
            text("SELECT 1 FROM pg_roles WHERE rolname = :name"), {"name": APP_ROLE}
        ).scalar()
    assert found == 1


def test_init_can_run_twice(app_password):
    init_database(get_admin_engine(), app_password)
    init_database(get_admin_engine(), app_password)

    assert missing_tables(get_admin_engine()) == []


def test_init_updates_the_password_of_an_existing_role(app_password):
    url = make_url(os.environ["DATABASE_URL"])
    new_password = 'it\'s a "new" password'
    init_database(get_admin_engine(), new_password)
    try:
        engine = create_engine(url.set(password=new_password))
        with engine.connect() as connection:
            assert connection.execute(text("SELECT 1")).scalar() == 1
        engine.dispose()
        old = create_engine(url.set(password=app_password))
        with pytest.raises(OperationalError):
            old.connect()
        old.dispose()
    finally:
        init_database(get_admin_engine(), app_password)


def test_role_can_insert_and_read_inputs_and_outputs():
    with get_engine().begin() as connection:
        input_id = insert_input(connection)
        connection.execute(
            text(
                "INSERT INTO model_outputs (input_id, score, alert, model_version) "
                "VALUES (:id, 0.3, true, 'test-0')"
            ),
            {"id": input_id},
        )
        assert connection.execute(text("SELECT count(*) FROM model_inputs")).scalar() == 1
        assert connection.execute(text("SELECT count(*) FROM model_outputs")).scalar() == 1


def test_role_can_insert_and_read_api_errors():
    with get_engine().begin() as connection:
        connection.execute(
            text(
                "INSERT INTO api_errors (method, path, status_code, detail) "
                "VALUES ('POST', '/predict', 401, '{}')"
            )
        )
        assert connection.execute(text("SELECT count(*) FROM api_errors")).scalar() == 1


def test_role_can_read_api_keys(api_key):
    with get_engine().connect() as connection:
        assert connection.execute(text("SELECT count(*) FROM api_keys")).scalar() == 1


@pytest.mark.parametrize(
    "statement",
    [
        "DELETE FROM model_inputs",
        "UPDATE model_inputs SET age = 30",
        "TRUNCATE model_inputs",
        "DROP TABLE model_outputs",
        "DELETE FROM api_errors",
        "UPDATE api_errors SET status_code = 200",
        "TRUNCATE api_errors",
        "DELETE FROM api_keys",
        "UPDATE api_keys SET client_name = 'x'",
        "INSERT INTO api_keys (client_name, key_hash) VALUES ('x', 'y')",
        "CREATE TABLE intruder (id integer)",
        "SELECT * FROM employees",
    ],
)
def test_role_is_refused_everything_else(statement):
    with get_engine().connect() as connection:
        with pytest.raises(ProgrammingError, match="permission denied|must be owner"):
            connection.execute(text(statement))


def test_missing_tables_lists_the_absent_ones(drop_all_tables):
    with get_admin_engine().begin() as connection:
        connection.execute(text("DROP TABLE model_outputs"))

    assert missing_tables(get_admin_engine()) == ["model_outputs"]


def test_api_refuses_to_start_when_a_table_is_missing(drop_all_tables):
    with get_admin_engine().begin() as connection:
        connection.execute(text("DROP TABLE model_outputs"))

    with pytest.raises(RuntimeError, match="Missing tables: model_outputs.*init"):
        with TestClient(app):
            pass


def test_cli_init_creates_the_first_key_once(capsys):
    assert main(["init"]) == 0
    first = capsys.readouterr().out
    assert "Initial key" in first
    key = first.strip().splitlines()[-1]
    assert len(key) >= 40

    assert main(["init"]) == 0
    second = capsys.readouterr().out
    assert key not in second
    assert "created" not in second
    assert main(["list-keys"]) == 0
    assert capsys.readouterr().out.count("Initial key") == 1


def test_cli_init_does_not_mint_a_key_when_only_a_revoked_one_exists(capsys):
    main(["create-key", "Old client"])
    main(["revoke-key", "1"])
    capsys.readouterr()

    assert main(["init"]) == 0

    assert "created" not in capsys.readouterr().out


def test_cli_init_needs_the_app_password(monkeypatch, capsys):
    monkeypatch.delenv("APP_DB_PASSWORD")

    assert main(["init"]) == 2
    assert "APP_DB_PASSWORD" in capsys.readouterr().err


def test_cli_needs_the_admin_url(monkeypatch):
    monkeypatch.delenv("DATABASE_ADMIN_URL")

    with pytest.raises(SystemExit) as exit_info:
        main(["list-keys"])

    assert exit_info.value.code == 2


def test_key_commands_do_not_need_the_app_password(monkeypatch, capsys):
    monkeypatch.delenv("APP_DB_PASSWORD")

    assert main(["create-key", "Client A"]) == 0


def test_create_key_prints_the_key_once_and_stores_its_hash(capsys):
    assert main(["create-key", "Client A"]) == 0
    key = capsys.readouterr().out.strip().splitlines()[-1]

    with get_admin_engine().connect() as connection:
        rows = connection.execute(text("SELECT client_name, key_hash FROM api_keys")).all()
    assert [row.client_name for row in rows] == ["Client A"]
    assert key not in rows[0].key_hash


def test_revoke_key_succeeds_then_reports_it_is_already_revoked(capsys):
    main(["create-key", "Client A"])
    capsys.readouterr()

    assert main(["revoke-key", "1"]) == 0
    assert "Key 1 revoked." in capsys.readouterr().out
    assert main(["revoke-key", "1"]) == 1
    assert "No active key" in capsys.readouterr().err


def test_revoke_key_with_an_unknown_id_returns_1(capsys):
    assert main(["revoke-key", "999"]) == 1
    assert "No active key with id 999" in capsys.readouterr().err


def test_list_keys_shows_status_and_never_a_hash(capsys):
    main(["create-key", "Client A"])
    main(["create-key", "Client B"])
    main(["revoke-key", "2"])
    capsys.readouterr()

    assert main(["list-keys"]) == 0
    output = capsys.readouterr().out

    assert "Client A" in output and "active" in output
    assert "Client B" in output and "revoked" in output
    with get_admin_engine().connect() as connection:
        hashes = connection.execute(text("SELECT key_hash FROM api_keys")).scalars().all()
    assert all(value not in output for value in hashes)


def test_key_commands_say_to_run_init_when_tables_are_missing(drop_all_tables, capsys):
    with get_admin_engine().begin() as connection:
        connection.execute(text("DROP TABLE model_outputs"))

    assert main(["list-keys"]) == 1
    assert "init" in capsys.readouterr().err
