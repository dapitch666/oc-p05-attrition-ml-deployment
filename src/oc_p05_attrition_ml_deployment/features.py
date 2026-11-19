import pandas as pd

TARGET = "a_quitte_l_entreprise"

SATISFACTION_COLUMNS = [
    "satisfaction_employee_environnement",
    "satisfaction_employee_equilibre_pro_perso",
    "satisfaction_employee_equipe",
    "satisfaction_employee_nature_travail",
]

FEATURE_COLUMNS = [
    "satisfaction_lows",
    "career_share_in_company",
    "role_share_of_tenure",
    "promotion_wait_share",
    "years_per_employer",
]

DROPS = {
    "niveau_hierarchique_poste": "Highly correlated with monthly income (0.95)",
    "annees_dans_le_poste_actuel": "Rebuilt from the role share and the years in the company",
    "annees_depuis_la_derniere_promotion": "Kept as the promotion wait share (0.91 in Spearman)",
    "note_evaluation_actuelle": "Two values only, redundant with the raise (0.77)",
    "departement": "Largely redundant with the job title",
}


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return the five engineered features of df, row by row.

    A ratio whose denominator is 0 takes a neutral value: 1 for the two shares of
    time already spent (a new hire has spent all of its tenure in the company and
    in the role), and 0 for the promotion wait.
    """
    experience = df["annees_experience_totale"]
    tenure = df["annees_dans_l_entreprise"]
    features = pd.DataFrame(index=df.index)
    features["satisfaction_lows"] = (df[SATISFACTION_COLUMNS] == 1).sum(axis=1).clip(upper=3)
    features["career_share_in_company"] = (tenure / experience).where(experience > 0, 1.0)
    features["role_share_of_tenure"] = (df["annees_dans_le_poste_actuel"] / tenure).where(
        tenure > 0, 1.0
    )
    features["promotion_wait_share"] = (df["annees_depuis_la_derniere_promotion"] / tenure).where(
        tenure > 0, 0.0
    )
    # 0 and 1 previous employers both describe a career spent in the company
    features["years_per_employer"] = experience / df["nombre_experiences_precedentes"].clip(lower=1)
    return features[FEATURE_COLUMNS]


def build_inputs(df: pd.DataFrame) -> pd.DataFrame:
    """Return the model inputs, not encoded yet, without the dropped columns and the target."""
    return df.drop(columns=[TARGET, *DROPS], errors="ignore").join(add_engineered_features(df))
