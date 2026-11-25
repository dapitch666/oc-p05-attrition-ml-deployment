import os
from collections.abc import Iterator

import pandas as pd
import pytest
from sqlalchemy import text
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session

from oc_p05_attrition_ml_deployment import models  # noqa: F401  # registers the tables
from oc_p05_attrition_ml_deployment.auth import create_api_key
from oc_p05_attrition_ml_deployment.db import (
    Base,
    create_database_if_missing,
    create_tables,
    get_engine,
)


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
        # Rebuild the tables so that the schema always matches the models.
        Base.metadata.drop_all(get_engine())
        create_tables(get_engine())
        yield
        get_engine().dispose()
        get_engine.cache_clear()


def truncate_model_tables() -> None:
    with get_engine().begin() as connection:
        connection.execute(
            text("TRUNCATE model_outputs, model_inputs, employees, api_keys RESTART IDENTITY")
        )


@pytest.fixture(autouse=True)
def empty_model_tables(test_database: None) -> Iterator[None]:
    """Leave the model tables empty before and after each test."""
    truncate_model_tables()
    yield
    truncate_model_tables()


@pytest.fixture
def api_key() -> str:
    """A valid key, created in the test database."""
    with Session(get_engine()) as session:
        return create_api_key(session, "Test client")[1]


@pytest.fixture
def employees_frame() -> pd.DataFrame:
    """Three invented employees, shaped like the project 4 dataset (id_employee index)."""
    rows = [
        {
            "id_employee": 7,
            "age": 29,
            "genre": "F",
            "revenu_mensuel": 3200,
            "statut_marital": "Célibataire",
            "departement": "Consulting",
            "poste": "Consultant",
            "nombre_experiences_precedentes": 1,
            "annees_experience_totale": 6,
            "annees_dans_l_entreprise": 3,
            "annees_dans_le_poste_actuel": 2,
            "satisfaction_employee_environnement": 2,
            "note_evaluation_precedente": 3,
            "niveau_hierarchique_poste": 1,
            "satisfaction_employee_nature_travail": 2,
            "satisfaction_employee_equipe": 3,
            "satisfaction_employee_equilibre_pro_perso": 1,
            "note_evaluation_actuelle": 3,
            "heures_supplementaires": "Oui",
            "augmentation_salaire_precedente": 11,
            "a_quitte_l_entreprise": "Oui",
            "nombre_participation_pee": 0,
            "nb_formations_suivies": 1,
            "distance_domicile_travail": 24,
            "niveau_education": 2,
            "domaine_etude": "Marketing",
            "frequence_deplacement": "Fréquent",
            "annees_depuis_la_derniere_promotion": 2,
            "annees_sous_responsable_actuel": 1,
        },
        {
            "id_employee": 12,
            "age": 45,
            "genre": "M",
            "revenu_mensuel": 9100,
            "statut_marital": "Marié(e)",
            "departement": "Commercial",
            "poste": "Manager",
            "nombre_experiences_precedentes": 3,
            "annees_experience_totale": 20,
            "annees_dans_l_entreprise": 12,
            "annees_dans_le_poste_actuel": 6,
            "satisfaction_employee_environnement": 4,
            "note_evaluation_precedente": 4,
            "niveau_hierarchique_poste": 3,
            "satisfaction_employee_nature_travail": 4,
            "satisfaction_employee_equipe": 4,
            "satisfaction_employee_equilibre_pro_perso": 3,
            "note_evaluation_actuelle": 4,
            "heures_supplementaires": "Non",
            "augmentation_salaire_precedente": 14,
            "a_quitte_l_entreprise": "Non",
            "nombre_participation_pee": 3,
            "nb_formations_suivies": 4,
            "distance_domicile_travail": 5,
            "niveau_education": 4,
            "domaine_etude": "Infra & Cloud",
            "frequence_deplacement": "Aucun",
            "annees_depuis_la_derniere_promotion": 4,
            "annees_sous_responsable_actuel": 5,
        },
        {
            "id_employee": 40,
            "age": 37,
            "genre": "F",
            "revenu_mensuel": 5400,
            "statut_marital": "Divorcé(e)",
            "departement": "Consulting",
            "poste": "Tech Lead",
            "nombre_experiences_precedentes": 2,
            "annees_experience_totale": 14,
            "annees_dans_l_entreprise": 7,
            "annees_dans_le_poste_actuel": 3,
            "satisfaction_employee_environnement": 3,
            "note_evaluation_precedente": 3,
            "niveau_hierarchique_poste": 2,
            "satisfaction_employee_nature_travail": 3,
            "satisfaction_employee_equipe": 2,
            "satisfaction_employee_equilibre_pro_perso": 3,
            "note_evaluation_actuelle": 3,
            "heures_supplementaires": "Non",
            "augmentation_salaire_precedente": 13,
            "a_quitte_l_entreprise": "Non",
            "nombre_participation_pee": 1,
            "nb_formations_suivies": 2,
            "distance_domicile_travail": 15,
            "niveau_education": 3,
            "domaine_etude": "Transformation Digitale",
            "frequence_deplacement": "Occasionnel",
            "annees_depuis_la_derniere_promotion": 1,
            "annees_sous_responsable_actuel": 2,
        },
    ]
    return pd.DataFrame(rows).set_index("id_employee")
