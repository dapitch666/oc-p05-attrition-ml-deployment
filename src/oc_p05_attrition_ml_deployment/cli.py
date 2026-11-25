"""Initialise the database and manage API keys; a key is shown once, only its hash is kept."""

import argparse
import os
import sys

from sqlalchemy import Engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from oc_p05_attrition_ml_deployment.auth import (
    create_api_key,
    has_api_keys,
    list_api_keys,
    revoke_api_key,
)
from oc_p05_attrition_ml_deployment.db import get_admin_engine, init_database, missing_tables

INITIAL_KEY_NAME = "Initial key"


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser with its subcommands."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("init", help="Create the tables, the application role and a first key.")
    create = commands.add_parser("create-key", help="Create a key for a client.")
    create.add_argument("client_name")
    revoke = commands.add_parser("revoke-key", help="Revoke a key by its id.")
    revoke.add_argument("key_id", type=int)
    commands.add_parser("list-keys", help="List the keys, without their hashes.")
    return parser


def print_new_key(key_id: int, client_name: str, key: str) -> None:
    """Print a new key and warn that it will not be shown again."""
    print(f"Key {key_id} created for {client_name}. It is shown once, store it now:")
    print(key)


def run_init(engine: Engine) -> int:
    """Initialise the database and create the first key when none exists."""
    password = os.environ.get("APP_DB_PASSWORD")
    if not password:
        print("APP_DB_PASSWORD is not set.", file=sys.stderr)
        return 2
    init_database(engine, password)
    print("Database initialised.")
    with Session(engine) as session:
        if not has_api_keys(session):
            print_new_key(*_create(session, INITIAL_KEY_NAME))
    return 0


def _create(session: Session, client_name: str) -> tuple[int, str, str]:
    key_id, key = create_api_key(session, client_name)
    return key_id, client_name, key


def run_key_command(engine: Engine, args: argparse.Namespace) -> int:
    """Run create-key, revoke-key or list-keys."""
    if missing := missing_tables(engine):
        print(f"Missing tables: {', '.join(missing)}. Run the init command first.", file=sys.stderr)
        return 1
    with Session(engine) as session:
        if args.command == "create-key":
            print_new_key(*_create(session, args.client_name))
            return 0
        if args.command == "list-keys":
            for record in list_api_keys(session):
                status = "revoked" if record.revoked_at else "active"
                print(f"{record.id}\t{record.client_name}\t{record.created_at:%Y-%m-%d}\t{status}")
            return 0
        if revoke_api_key(session, args.key_id):
            print(f"Key {args.key_id} revoked.")
            return 0
    print(f"No active key with id {args.key_id}.", file=sys.stderr)
    return 1


def main(argv: list[str] | None = None) -> int:
    """Run the command line and return the exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if not os.environ.get("DATABASE_ADMIN_URL"):
        parser.error("DATABASE_ADMIN_URL is not set.")
    engine = get_admin_engine()
    try:
        if args.command == "init":
            return run_init(engine)
        return run_key_command(engine, args)
    except SQLAlchemyError as error:
        print(f"Database error: {type(error).__name__}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
