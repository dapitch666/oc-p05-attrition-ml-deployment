import os
from collections.abc import Iterator

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url

from oc_p05_attrition_ml_deployment import models  # noqa: F401  # registers the tables
from oc_p05_attrition_ml_deployment.db import create_tables, get_engine


def create_database_if_missing(url: str) -> None:
    """Create the database named in the URL, going through the maintenance database."""
    parsed = make_url(url)
    admin = create_engine(parsed.set(database="postgres"), isolation_level="AUTOCOMMIT")
    try:
        with admin.connect() as connection:
            exists = connection.execute(
                text("SELECT 1 FROM pg_database WHERE datname = :name"), {"name": parsed.database}
            ).scalar()
            if not exists:
                name = admin.dialect.identifier_preparer.quote(parsed.database)
                connection.execute(text(f"CREATE DATABASE {name}"))
    finally:
        admin.dispose()


@pytest.fixture(scope="session", autouse=True)
def test_database() -> Iterator[None]:
    """Run the whole session against TEST_DATABASE_URL, never against the demo database."""
    url = os.environ.get("TEST_DATABASE_URL")
    if not url:
        pytest.exit("TEST_DATABASE_URL is not set.", returncode=2)
    if not (make_url(url).database or "").endswith("_test"):
        pytest.exit("The database name in TEST_DATABASE_URL must end with _test.", returncode=2)
    create_database_if_missing(url)
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("DATABASE_URL", url)
        get_engine.cache_clear()
        create_tables(get_engine())
        yield
        get_engine().dispose()
        get_engine.cache_clear()


def truncate_prediction_logs() -> None:
    with get_engine().begin() as connection:
        connection.execute(text("TRUNCATE prediction_logs RESTART IDENTITY"))


@pytest.fixture(autouse=True)
def empty_prediction_logs(test_database: None) -> Iterator[None]:
    """Leave prediction_logs empty before and after each test."""
    truncate_prediction_logs()
    yield
    truncate_prediction_logs()
