from collections.abc import Iterator

import pandas as pd
import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from oc_p05_attrition_ml_deployment.dataset import (
    EXTRA_COLUMNS,
    DatasetError,
    build_employees,
    import_employees,
)
from oc_p05_attrition_ml_deployment.db import get_admin_engine
from oc_p05_attrition_ml_deployment.features import TARGET
from oc_p05_attrition_ml_deployment.models import Employee
from oc_p05_attrition_ml_deployment.schemas import EmployeeInput


@pytest.fixture
def session() -> Iterator[Session]:
    with Session(get_admin_engine()) as session:
        yield session


def count_employees(session: Session) -> int:
    return session.scalar(select(func.count()).select_from(Employee))


def test_import_inserts_every_row_with_its_identifier(session, employees_frame):
    assert import_employees(session, employees_frame) == 3

    stored = {e.id_employee: e for e in session.scalars(select(Employee))}
    assert set(stored) == {7, 12, 40}
    assert stored[7].a_quitte_l_entreprise is True
    assert stored[12].a_quitte_l_entreprise is False
    assert stored[40].statut_marital == "Divorcé(e)"
    assert stored[12].departement == "Commercial"
    assert stored[12].niveau_hierarchique_poste == 3
    assert stored[7].note_evaluation_actuelle == 3
    assert stored[7].revenu_mensuel == 3200


def test_second_import_inserts_nothing(session, employees_frame):
    import_employees(session, employees_frame)

    assert import_employees(session, employees_frame) == 0
    assert count_employees(session) == 3


def test_invalid_row_is_reported_by_id_without_its_values(session, employees_frame):
    employees_frame.loc[12, "age"] = 17
    employees_frame.loc[12, "revenu_mensuel"] = 9100

    with pytest.raises(DatasetError) as error:
        import_employees(session, employees_frame)

    message = str(error.value)
    assert "id_employee 12" in message
    assert "age" in message
    assert "17" not in message
    assert "9100" not in message
    assert count_employees(session) == 0


def test_incoherent_years_are_reported_by_id(employees_frame):
    employees_frame.loc[7, "annees_dans_l_entreprise"] = 30

    with pytest.raises(DatasetError, match="id_employee 7"):
        build_employees(employees_frame)


def test_invalid_target_is_rejected_without_its_value(employees_frame):
    employees_frame.loc[40, TARGET] = "Peut-etre"

    with pytest.raises(DatasetError) as error:
        build_employees(employees_frame)

    assert f"id_employee 40: {TARGET}" in str(error.value)
    assert "Peut-etre" not in str(error.value)


def test_error_report_is_capped(employees_frame):
    many = pd.concat([employees_frame.iloc[[0]].rename(index={7: i}) for i in range(100, 130)])
    many["age"] = 17

    with pytest.raises(DatasetError) as error:
        build_employees(many)

    assert "and 10 more" in str(error.value)


def test_index_must_be_named_id_employee(employees_frame):
    with pytest.raises(DatasetError, match="named id_employee"):
        build_employees(employees_frame.rename_axis("id"))


def test_duplicate_identifiers_are_rejected(employees_frame):
    employees_frame.index = [7, 7, 40]
    employees_frame.index.name = "id_employee"

    with pytest.raises(DatasetError, match="unique"):
        build_employees(employees_frame)


def test_missing_column_is_rejected(employees_frame):
    with pytest.raises(DatasetError, match="departement"):
        build_employees(employees_frame.drop(columns="departement"))


def test_null_value_is_rejected_by_column_name(employees_frame):
    employees_frame.loc[7, "note_evaluation_actuelle"] = None

    with pytest.raises(DatasetError, match="note_evaluation_actuelle"):
        build_employees(employees_frame)


def test_empty_dataset_is_rejected(employees_frame):
    with pytest.raises(DatasetError, match="no row"):
        build_employees(employees_frame.iloc[0:0])


def test_employee_columns_match_the_dataset_fields():
    columns = {column.name for column in Employee.__table__.columns}

    assert columns == {*EmployeeInput.model_fields, *EXTRA_COLUMNS, TARGET, "id_employee"}
