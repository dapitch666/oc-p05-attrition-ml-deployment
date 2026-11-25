"""Create or revoke API keys; a created key is shown once and only its hash is stored."""

import argparse
import os
import sys

from sqlalchemy.orm import Session

from oc_p05_attrition_ml_deployment.auth import create_api_key, revoke_api_key
from oc_p05_attrition_ml_deployment.db import create_tables, get_engine


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("create", help="Create a key for a client.")
    create.add_argument("client_name")
    revoke = commands.add_parser("revoke", help="Revoke a key by its id.")
    revoke.add_argument("key_id", type=int)
    args = parser.parse_args()

    if not os.environ.get("DATABASE_URL"):
        parser.error("DATABASE_URL is not set.")
    engine = get_engine()
    create_tables(engine)
    with Session(engine) as session:
        if args.command == "create":
            key_id, key = create_api_key(session, args.client_name)
            print(f"Key {key_id} created for {args.client_name}. It is shown once:")
            print(key)
            return 0
        if revoke_api_key(session, args.key_id):
            print(f"Key {args.key_id} revoked.")
            return 0
    print(f"No active key with id {args.key_id}.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
