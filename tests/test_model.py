import pandas as pd
from sklearn.linear_model import LogisticRegression

from oc_p05_attrition_ml_deployment.model import (
    BINARY_COLUMNS,
    NOMINAL_COLUMNS,
    NUMERIC_COLUMNS,
    PIPELINE_COLUMNS,
    TRAVEL_COLUMNS,
    build_pipeline,
)


def make_frame() -> pd.DataFrame:
    rows = 6
    data = {column: [float(i) for i in range(rows)] for column in NUMERIC_COLUMNS}
    data[TRAVEL_COLUMNS[0]] = ["Aucun", "Occasionnel", "Fréquent"] * 2
    data[BINARY_COLUMNS[0]] = ["F", "M"] * 3
    data[BINARY_COLUMNS[1]] = ["Oui", "Non", "Non"] * 2
    for i, column in enumerate(NOMINAL_COLUMNS):
        data[column] = [f"{column}_{(j + i) % 3}" for j in range(rows)]
    return pd.DataFrame(data)[PIPELINE_COLUMNS]


def test_pipeline_returns_one_probability_per_row():
    X = make_frame()
    y = [0, 1, 0, 1, 0, 1]
    pipeline = build_pipeline(LogisticRegression()).fit(X, y)
    proba = pipeline.predict_proba(X)[:, 1]
    assert len(proba) == len(X)
    assert ((proba >= 0) & (proba <= 1)).all()
