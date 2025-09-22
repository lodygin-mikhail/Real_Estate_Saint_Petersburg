import json
from typing import List

import click
import mlflow
import pandas as pd
import joblib as jb
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

mlflow.set_experiment("Real Estate Evaluation")


@click.command()
@click.argument("input_paths", type=click.Path(exists=True), nargs=2)
@click.argument("output_path", type=click.Path())
def evaluate(input_paths: List[str], output_path: str):
    test_df = pd.read_csv(input_paths[0])
    model = jb.load(input_paths[1])

    x_test = test_df.drop("price", axis=1)
    y_test = test_df["price"]

    y_pred = model.predict(x_test)
    score = dict(
        rmse=root_mean_squared_error(y_test, y_pred),
        mae=mean_absolute_error(y_test, y_pred),
        r2=r2_score(y_test, y_pred),
    )
    mlflow.log_metrics(score)

    with open(output_path, "w") as score_file:
        json.dump(score, score_file, indent=4)


if __name__ == "__main__":
    evaluate()
