from typing import List

import pandas as pd
import click
import joblib as jb
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

RANDOM_STATE = 42

NUM_COLS = [
    "total_area_m2",
    "living_area_m2",
    "kitchen_area_m2",
    "floor",
    "minutes_to_metro",
    "house_age",
]
CAT_COLS = [
    "flat_status",
    "num_of_rooms",
    "metro_station",
    "transfer_type",
    "is_future_building",
]


@click.command()
@click.argument("input_paths", type=click.Path(exists=True), nargs=2)
@click.argument("output_path", type=click.Path())
def train(input_paths: List[str], output_path: str):
    train_df = pd.read_csv(input_paths[0])
    test_df = pd.read_csv(input_paths[1])

    x_train = train_df.drop("price", axis=1)
    y_train = train_df["price"]
    x_test = test_df.drop("price", axis=1)
    y_test = test_df["price"]

    data_preprocessor = ColumnTransformer(
        [
            (
                "ohe",
                OneHotEncoder(
                    drop="first", handle_unknown="ignore", sparse_output=False
                ),
                CAT_COLS,
            ),
            ("num", StandardScaler(), NUM_COLS),
        ],
        remainder="passthrough",
    )

    pipeline = Pipeline(
        [("preprocessor", data_preprocessor), ("models", LinearRegression())]
    )

    param_grid = [
        {
            "models": [DecisionTreeRegressor(random_state=RANDOM_STATE)],
            "models__max_depth": [1, 2, 3],
            "models__min_samples_split": [2, 3],
        },
        {
            "models": [RandomForestRegressor(random_state=RANDOM_STATE)],
            "models__max_depth": [1, 2, 3],
            "models__min_samples_split": [2, 3],
        },
        {"models": [LGBMRegressor(verbose=-1, random_state=RANDOM_STATE)]},
        {"models": [CatBoostRegressor(verbose=-1, random_state=RANDOM_STATE)]},
    ]

    gs = GridSearchCV(
        pipeline, param_grid, scoring="neg_root_mean_squared_error", cv=5, n_jobs=-1
    )

    gs.fit(x_train, y_train)

    jb.dump(gs.best_estimator_, output_path)


if __name__ == "__main__":
    train()
