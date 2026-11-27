import os
from collections.abc import Generator
from functools import lru_cache

from psycopg import Cursor, sql
from sqlalchemy import Engine, create_engine, inspect, text
from sqlalchemy.engine import make_url
from sqlalchemy.orm import DeclarativeBase, Session

CONNECT_TIMEOUT_SECONDS = 3
APP_ROLE = "attrition_app"
REQUIRED_TABLES = ("api_keys", "model_inputs", "model_outputs", "api_errors")
# Tables the application role may write; api_keys is read only, the other tables are closed to it.
APP_INSERT_TABLES = ("model_inputs", "model_outputs", "api_errors")


class Base(DeclarativeBase):
    """Declarative base of the ORM models."""


def _build_engine(variable: str) -> Engine:
    """Build an engine from the URL held in an environment variable."""
    url = os.environ.get(variable)
    if not url:
        raise RuntimeError(f"{variable} is not set.")
    return create_engine(
        url,
        pool_pre_ping=True,
        # Keep the employee fields out of the logs: error messages would list the parameters.
        hide_parameters=True,
        # Fail fast when PostgreSQL is unreachable, so the API can answer 503 quickly.
        connect_args={"connect_timeout": CONNECT_TIMEOUT_SECONDS},
    )


@lru_cache
def get_engine() -> Engine:
    """Build the application engine from DATABASE_URL, once; fail if it is missing."""
    return _build_engine("DATABASE_URL")


@lru_cache
def get_admin_engine() -> Engine:
    """Build the owner engine from DATABASE_ADMIN_URL, once; fail if it is missing."""
    return _build_engine("DATABASE_ADMIN_URL")


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


def create_tables(engine: Engine) -> None:
    """Create the tables that do not exist yet."""
    Base.metadata.create_all(engine)


def init_database(engine: Engine, app_password: str) -> None:
    """Create the tables and the application role with its limited rights; safe to repeat."""
    create_tables(engine)
    role = sql.Identifier(APP_ROLE)
    password = sql.Literal(app_password)
    # psycopg 3 binds parameters server side, which CREATE ROLE does not accept: build the text.
    with engine.begin() as connection:
        cursor = connection.connection.cursor()
        cursor.execute("SELECT 1 FROM pg_roles WHERE rolname = %s", (APP_ROLE,))
        verb = "ALTER" if cursor.fetchone() else "CREATE"
        cursor.execute(
            sql.SQL("{} ROLE {} LOGIN PASSWORD {}").format(sql.SQL(verb), role, password)
        )
        cursor.execute("SELECT current_database()")
        database = sql.Identifier(cursor.fetchone()[0])

        for table in Base.metadata.sorted_tables:
            cursor.execute(
                sql.SQL("REVOKE ALL ON TABLE {} FROM {}").format(sql.Identifier(table.name), role)
            )
            for sequence in _sequences(cursor, table.name):
                cursor.execute(sql.SQL("REVOKE ALL ON SEQUENCE {} FROM {}").format(sequence, role))

        cursor.execute(sql.SQL("GRANT CONNECT ON DATABASE {} TO {}").format(database, role))
        cursor.execute(sql.SQL("GRANT USAGE ON SCHEMA public TO {}").format(role))
        cursor.execute(sql.SQL("GRANT SELECT ON TABLE api_keys TO {}").format(role))
        for name in APP_INSERT_TABLES:
            cursor.execute(
                sql.SQL("GRANT SELECT, INSERT ON TABLE {} TO {}").format(sql.Identifier(name), role)
            )
            for sequence in _sequences(cursor, name):
                cursor.execute(sql.SQL("GRANT USAGE ON SEQUENCE {} TO {}").format(sequence, role))


def _sequences(cursor: Cursor, table: str) -> list[sql.SQL]:
    """Return the sequences owned by the columns of a table, already quoted by PostgreSQL."""
    cursor.execute(
        "SELECT pg_get_serial_sequence(quote_ident(%s), a.attname) "
        "FROM pg_attribute a WHERE a.attrelid = quote_ident(%s)::regclass "
        "AND a.attnum > 0 AND NOT a.attisdropped",
        (table, table),
    )
    # pg_get_serial_sequence returns NULL for a column without a sequence: skip it.
    return [sql.SQL(row[0]) for row in cursor.fetchall() if row[0]]


def missing_tables(engine: Engine) -> list[str]:
    """Return the required tables that are absent from the database."""
    present = set(inspect(engine).get_table_names())
    return [name for name in REQUIRED_TABLES if name not in present]


def get_session() -> Generator[Session]:
    """Yield a session for one request, then close it."""
    with Session(get_engine()) as session:
        yield session
