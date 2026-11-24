"""Create the database and tables, then import the project 4 dataset into employees."""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
from sqlalchemy.orm import Session

from oc_p05_attrition_ml_deployment.dataset import DatasetError, import_employees
from oc_p05_attrition_ml_deployment.db import create_database_if_missing, create_tables, get_engine

DEFAULT_DATASET = Path("data/employees_cleaned.parquet")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATASET, help="Parquet dataset.")
    args = parser.parse_args()

    url = os.environ.get("DATABASE_URL")
    if not url:
        parser.error("DATABASE_URL is not set.")
    create_database_if_missing(url)
    engine = get_engine()
    create_tables(engine)

    frame = pd.read_parquet(args.data)
    try:
        with Session(engine) as session:
            inserted = import_employees(session, frame)
    except DatasetError as error:
        print(error, file=sys.stderr)
        return 1
    if inserted:
        print(f"Imported {inserted} employees.")
    else:
        print("The employees table already holds rows: nothing imported.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
