from typing import List

import click
import pandas as pd
import joblib as jb
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    root_mean_squared_error,
)


@click.command()
@click.argument("input_paths", type=click.Path(exists=True), nargs=2)
@click.argument("output_path", type=click.Path())
def evaluate(input_paths: List[str], output_path: str):
    test_df = pd.read_csv(input_paths[0])
    model = jb.load(input_paths[1])

    x_test = test_df.drop("price", axis=1)
    y_test = test_df["price"]

    y_pred = model.predict(x_test)
    score = pd.DataFrame(
        dict(
            mae=mean_absolute_error(y_test, y_pred),
            rmse=root_mean_squared_error(y_test, y_pred),
        ),
        index=[0],
    )

    score.to_csv(output_path, index=False)


if __name__ == "__main__":
    evaluate()
