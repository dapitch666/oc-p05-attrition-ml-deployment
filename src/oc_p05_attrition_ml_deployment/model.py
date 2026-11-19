from sklearn.base import ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

from oc_p05_attrition_ml_deployment.features import FEATURE_COLUMNS

NUMERIC_COLUMNS = [
    "age",
    "revenu_mensuel",
    "nombre_experiences_precedentes",
    "annees_experience_totale",
    "annees_dans_l_entreprise",
    "satisfaction_employee_environnement",
    "note_evaluation_precedente",
    "satisfaction_employee_nature_travail",
    "satisfaction_employee_equipe",
    "satisfaction_employee_equilibre_pro_perso",
    "augmentation_salaire_precedente",
    "nombre_participation_pee",
    "nb_formations_suivies",
    "distance_domicile_travail",
    "niveau_education",
    "annees_sous_responsable_actuel",
    *FEATURE_COLUMNS,
]
TRAVEL_COLUMNS = ["frequence_deplacement"]
TRAVEL_CATEGORIES = ["Aucun", "Occasionnel", "Fréquent"]
BINARY_COLUMNS = ["genre", "heures_supplementaires"]
NOMINAL_COLUMNS = ["statut_marital", "poste", "domaine_etude"]

PIPELINE_COLUMNS = NUMERIC_COLUMNS + TRAVEL_COLUMNS + BINARY_COLUMNS + NOMINAL_COLUMNS


def build_pipeline(model: ClassifierMixin) -> Pipeline:
    """Return the pipeline with the steps "preprocessing" and "model"."""
    preprocessing = ColumnTransformer(
        [
            ("numeric", "passthrough", NUMERIC_COLUMNS),
            ("travel", OrdinalEncoder(categories=[TRAVEL_CATEGORIES]), TRAVEL_COLUMNS),
            ("binary", OneHotEncoder(drop="if_binary"), BINARY_COLUMNS),
            ("nominal", OneHotEncoder(handle_unknown="ignore"), NOMINAL_COLUMNS),
        ],
        verbose_feature_names_out=False,
    )
    return Pipeline([("preprocessing", preprocessing), ("model", model)])
