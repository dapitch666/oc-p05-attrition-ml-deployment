import numpy as np
import pandas as pd
import pytest

from oc_p05_attrition_ml_deployment.features import (
    DROPS,
    FEATURE_COLUMNS,
    SATISFACTION_COLUMNS,
    TARGET,
    add_engineered_features,
    build_inputs,
)

DEFAULT_ROW = {
    "annees_experience_totale": 10,
    "annees_dans_l_entreprise": 5,
    "annees_dans_le_poste_actuel": 2,
    "annees_depuis_la_derniere_promotion": 1,
    "nombre_experiences_precedentes": 2,
    "satisfaction_employee_environnement": 3,
    "satisfaction_employee_equilibre_pro_perso": 3,
    "satisfaction_employee_equipe": 3,
    "satisfaction_employee_nature_travail": 3,
}


def make_df(*overrides: dict) -> pd.DataFrame:
    return pd.DataFrame([{**DEFAULT_ROW, **row} for row in overrides])


def test_zero_experience_gives_full_career_share():
    features = add_engineered_features(
        make_df({"annees_experience_totale": 0, "annees_dans_l_entreprise": 0})
    )
    assert features["career_share_in_company"].iloc[0] == 1


def test_zero_tenure_gives_neutral_shares():
    features = add_engineered_features(
        make_df(
            {
                "annees_dans_l_entreprise": 0,
                "annees_dans_le_poste_actuel": 0,
                "annees_depuis_la_derniere_promotion": 0,
            }
        )
    )
    assert features["role_share_of_tenure"].iloc[0] == 1
    assert features["promotion_wait_share"].iloc[0] == 0


@pytest.mark.parametrize("employers", [0, 1])
def test_few_previous_employers_keep_total_experience(employers):
    features = add_engineered_features(
        make_df({"annees_experience_totale": 8, "nombre_experiences_precedentes": employers})
    )
    assert features["years_per_employer"].iloc[0] == 8


def test_years_per_employer_divides_by_employers():
    features = add_engineered_features(
        make_df({"annees_experience_totale": 8, "nombre_experiences_precedentes": 4})
    )
    assert features["years_per_employer"].iloc[0] == 2


def test_satisfaction_lows_are_capped_at_three():
    scores = dict.fromkeys(SATISFACTION_COLUMNS, 1)
    features = add_engineered_features(make_df(scores))
    assert features["satisfaction_lows"].iloc[0] == 3


def test_satisfaction_lows_count_below_cap():
    features = add_engineered_features(
        make_df({"satisfaction_employee_equipe": 1, "satisfaction_employee_environnement": 1})
    )
    assert features["satisfaction_lows"].iloc[0] == 2


def test_features_have_no_missing_or_infinite_value():
    df = make_df(
        {},
        {"annees_experience_totale": 0, "annees_dans_l_entreprise": 0},
        {"annees_dans_l_entreprise": 0, "nombre_experiences_precedentes": 0},
    )
    features = add_engineered_features(df)
    assert list(features.columns) == FEATURE_COLUMNS
    assert features.notna().all().all()
    assert np.isfinite(features).all().all()


def test_features_do_not_depend_on_other_rows():
    df = make_df({}, {"annees_experience_totale": 30, "annees_dans_l_entreprise": 0})
    together = add_engineered_features(df)
    alone = add_engineered_features(df.iloc[[0]])
    pd.testing.assert_frame_equal(alone, together.iloc[[0]])


def make_raw_df(with_target: bool) -> pd.DataFrame:
    df = make_df({}, {"annees_experience_totale": 4})
    for column in DROPS:
        if column not in df.columns:
            df[column] = 1
    df["age"] = [30, 40]
    if with_target:
        df[TARGET] = ["Oui", "Non"]
    return df


@pytest.mark.parametrize("with_target", [True, False])
def test_build_inputs_drops_columns_and_adds_features(with_target):
    inputs = build_inputs(make_raw_df(with_target))
    assert not set(DROPS) & set(inputs.columns)
    assert TARGET not in inputs.columns
    assert set(FEATURE_COLUMNS) <= set(inputs.columns)
    assert "age" in inputs.columns
    assert len(inputs) == 2


def test_build_inputs_does_not_modify_its_argument():
    df = make_raw_df(with_target=True)
    before = df.copy()
    build_inputs(df)
    pd.testing.assert_frame_equal(df, before)
