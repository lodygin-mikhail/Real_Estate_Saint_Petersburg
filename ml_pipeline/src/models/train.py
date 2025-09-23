import json
import os
from typing import List

import pandas as pd
import click
import joblib as jb
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
import mlflow
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor
from dotenv import load_dotenv

RANDOM_STATE = 42

load_dotenv()

os.environ.update(
    {
        "AWS_ACCESS_KEY_ID": os.getenv("MLFLOW_AWS_ACCESS_KEY_ID", ""),
        "AWS_SECRET_ACCESS_KEY": os.getenv("MLFLOW_AWS_SECRET_ACCESS_KEY", ""),
        "MLFLOW_S3_ENDPOINT_URL": os.getenv(
            "MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000"
        ),
    }
)

# Настройка MLflow
tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
if tracking_uri:
    mlflow.set_tracking_uri(tracking_uri)

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
@click.argument("output_path", type=click.Path(), nargs=2)
def train(input_paths: List[str], output_path: List[str]):
    """
    Function trains model and logges params, model and metrics with MLflow
    :param input_paths: Path to read train and test DataFrames
    :param output_path: Path to save trained model and metrics
    :return:
    """

    mlflow.set_experiment("Real Estate Price Prediction")

    with mlflow.start_run():
        train_df = pd.read_csv(input_paths[0])
        test_df = pd.read_csv(input_paths[1])

        # Логирование информации о данных
        mlflow.log_param("train_samples", len(train_df))
        mlflow.log_param("test_samples", len(test_df))
        mlflow.log_param("features", f"NUM: {NUM_COLS}, CAT: {CAT_COLS}")
        # Разделение данных на переменные и целевую переменную
        x_train = train_df.drop("price", axis=1)
        y_train = train_df["price"]
        x_test = test_df.drop("price", axis=1)
        y_test = test_df["price"]
        # Конфигурируем пайплайн обработки признаков
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
        # Задаем параметры сетки для GridSearchCV
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
        # Инициализируем обучение пайплайна
        gs.fit(x_train, y_train)

        # Сохранение модели
        jb.dump(gs.best_estimator_, output_path[0])

        # Логирование лучших параметров
        best_params = gs.best_params_
        mlflow.log_params(best_params)
        mlflow.log_param(
            "best_estimator", type(gs.best_estimator_.named_steps["models"]).__name__
        )

        # Предсказания и метрики
        y_pred = gs.best_estimator_.predict(x_test)

        # Логирование метрик
        score = dict(
            rmse=root_mean_squared_error(y_test, y_pred),
            mae=mean_absolute_error(y_test, y_pred),
            r2=r2_score(y_test, y_pred),
        )

        with open(output_path[1], "w") as score_file:
            json.dump(score, score_file, indent=4)

        mlflow.log_metrics(score)

        # Логирование модели в MLflow
        signature = infer_signature(x_train, y_pred)

        # Для sklearn моделей
        if hasattr(gs.best_estimator_, "predict"):
            mlflow.sklearn.log_model(
                gs.best_estimator_,
                name="model",
                signature=signature,
                registered_model_name="house_price_prediction",
            )

        # Логирование информации о feature importance (если доступно)
        try:
            if hasattr(
                gs.best_estimator_.named_steps["models"], "feature_importances_"
            ):
                feature_importances = gs.best_estimator_.named_steps[
                    "models"
                ].feature_importances_
                importance_dict = dict(zip(x_train.columns, feature_importances))
                mlflow.log_dict(importance_dict, "feature_importances.json")
        except Exception:
            pass


if __name__ == "__main__":
    train()
