import os
from collections.abc import Iterator
from functools import lru_cache

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import DeclarativeBase, Session


class Base(DeclarativeBase):
    """Declarative base of the ORM models."""


@lru_cache
def get_engine() -> Engine:
    """Build the engine from DATABASE_URL, once; fail if the variable is missing."""
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError("DATABASE_URL is not set.")
    return create_engine(url, pool_pre_ping=True)


def create_tables(engine: Engine) -> None:
    """Create the tables that do not exist yet."""
    Base.metadata.create_all(engine)


def get_session() -> Iterator[Session]:
    """Yield a session for one request, then close it."""
    with Session(get_engine()) as session:
        yield session
