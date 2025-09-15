from ast import literal_eval
from datetime import datetime
from pathlib import Path
from typing import List

import click
import pandas as pd

INTERIM_DATA_DIR = Path(__file__).resolve().parent.parent.parent / 'data' / 'interim'

def nearest_metro_station(json: dict) -> List[str]:
    """
    Function chooses the nearest metro station from the json with its time and type of transport
    :param json: dict with three nearest metro stations and its time and type of transport
    :return: List of metro nearest metro station, time to get there and type of transport
    """
    stations_dict = literal_eval(json)

    nearest_station = stations_dict['Underground name'][0]
    time_to_get = stations_dict['Underground time'][0]
    way_to_get_to = stations_dict['Transport type'][0]

    return [nearest_station, time_to_get, way_to_get_to]

@click.command()
@click.argument('input_path', type=click.Path())
@click.argument('output_path', type=click.Path())
def add_features(input_path: str, output_path: str):
    """ This function adds features to a csv file.
    :param input_path:
    :param output_path:
    :return:
    """
    df = pd.read_csv(input_path)
    # Применяем функцию для определения ближайшей станции метро
    df[['metro_station', 'minutes_to_metro', 'transfer_type']] = pd.DataFrame(df['metro_info'].apply(
        lambda x: nearest_metro_station(x)).tolist(),index=df.index)
    # Избавляемся от общей колонки
    df.drop('metro_info', axis=1, inplace=True)
    # Изменим значения в столбце transfer_type для лучшей читаемости
    df['transfer_type'] = df['transfer_type'].replace({'car': 'на машине', 'legs': 'пешком'})
    df['minutes_to_metro'] = df['minutes_to_metro'].replace('откроется', 10).astype(int)
    # Добавим новый признак возраст дома
    current_year = datetime.now().year
    df['house_age'] = (current_year - df['year_of_bulding']).apply(lambda x: x if x > 0 else 0)
    # Добавим бинарный признак строится ли еще дом
    df['is_future_building'] = (df['year_of_bulding'] > current_year).astype(int)
    df.drop('year_of_bulding', axis=1, inplace=True)

    df.to_csv(output_path, index=False)

if __name__ == '__main__':
    add_features()