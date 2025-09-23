import click
import pandas as pd
from typing import List


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_file_paths", type=click.Path(), nargs=2)
def prepare_datasets(input_filepath, output_file_paths: List[str]):
    """
    Function splits data into train and test sets
    :param input_filepath: Path to read prepared DataFrame
    :param output_file_paths: Path to save train and test sets
    :return:
    """
    df = pd.read_csv(input_filepath)
    df = df.drop_duplicates()
    # Делим датафрейм на трейн и тест в пропорции 75:25
    train = df.sample(frac=0.75, random_state=42)
    test = df.drop(train.index)
    # Сохраняем полученные сеты
    train.to_csv(output_file_paths[0], index=False)
    test.to_csv(output_file_paths[1], index=False)


if __name__ == "__main__":
    prepare_datasets()
