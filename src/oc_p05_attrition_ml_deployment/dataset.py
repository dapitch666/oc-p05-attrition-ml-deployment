import pandas as pd
from pydantic import ValidationError
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from oc_p05_attrition_ml_deployment.features import TARGET
from oc_p05_attrition_ml_deployment.models import Employee
from oc_p05_attrition_ml_deployment.schemas import EmployeeInput

INDEX_NAME = "id_employee"
EXTRA_COLUMNS = ("departement", "niveau_hierarchique_poste", "note_evaluation_actuelle")
TARGET_VALUES = {"Oui": True, "Non": False}
MAX_REPORTED_PROBLEMS = 20


class DatasetError(Exception):
    """The dataset cannot be imported; the message names id_employee and fields, never values."""


def build_employees(frame: pd.DataFrame) -> list[Employee]:
    """Validate every row against EmployeeInput and the target values, and build the rows."""
    if frame.index.name != INDEX_NAME:
        raise DatasetError(f"The index of the dataset must be named {INDEX_NAME}.")
    if frame.empty:
        raise DatasetError("The dataset has no row.")
    if not pd.api.types.is_integer_dtype(frame.index) or not frame.index.is_unique:
        raise DatasetError(f"{INDEX_NAME} must hold unique integers.")
    fields = list(EmployeeInput.model_fields)
    required = [*fields, *EXTRA_COLUMNS, TARGET]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise DatasetError(f"Missing columns: {', '.join(missing)}.")
    with_nulls = [column for column in required if frame[column].isna().any()]
    if with_nulls:
        raise DatasetError(f"Columns with null values: {', '.join(with_nulls)}.")

    employees: list[Employee] = []
    problems: list[str] = []
    for id_employee, row in zip(frame.index, frame[required].to_dict("records"), strict=True):
        try:
            EmployeeInput(**{name: row[name] for name in fields})
        except ValidationError as error:
            for detail in error.errors():
                where = ".".join(str(part) for part in detail["loc"]) or "years coherence"
                problems.append(f"{INDEX_NAME} {id_employee}: {where} ({detail['type']})")
            continue
        if row[TARGET] not in TARGET_VALUES:
            problems.append(f"{INDEX_NAME} {id_employee}: {TARGET} (invalid_value)")
            continue
        employees.append(
            Employee(**{**row, TARGET: TARGET_VALUES[row[TARGET]]}, id_employee=int(id_employee))
        )
    if problems:
        shown = problems[:MAX_REPORTED_PROBLEMS]
        rest = len(problems) - len(shown)
        suffix = f" ... and {rest} more." if rest else ""
        raise DatasetError("Invalid rows: " + "; ".join(shown) + suffix)
    return employees


def import_employees(session: Session, frame: pd.DataFrame) -> int:
    """Insert the whole frame in one transaction; return 0 if the table already has rows."""
    if session.scalar(select(func.count()).select_from(Employee)):
        return 0
    employees = build_employees(frame)
    session.add_all(employees)
    session.commit()
    return len(employees)
