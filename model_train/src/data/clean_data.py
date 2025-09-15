import click
import pandas as pd

MAX_PRICE = 45000000

@click.command()
@click.argument('input_path', type=click.Path())
@click.argument('output_path', type=click.Path())
def clean_data(input_path: str, output_path: str):
    """
    Function cleans data
    :param input_path:
    :param output_path:
    :return:
    """
    df = pd.read_csv(input_path)
    # Приводим названия колонок в snake_case
    df.columns = ['metro_info', 'flat_status', 'num_of_rooms', 'total_area_m2', 'living_area_m2', 'kitchen_area_m2',
                  'floor', 'year_of_bulding', 'price']
    # Заменяем выбивающиеся значения на заглушку Неизвестно
    df['num_of_rooms'] = df['num_of_rooms'].apply(lambda x: 'Неизвестно' if x in ['Многокомнатны',
                                                                                  'Многокомнатна',
                                                                                  '15/4',
                                                                                  'Квартир',
                                                                                  '29/4'] else x)
    # Объединяем различные написания квартир студий в одно
    df['num_of_rooms'] = df['num_of_rooms'].replace('Апартаменты-студия', 'Студия')
    # Добавляем окончание слов для лучшей читаемости
    df['num_of_rooms'] = df['num_of_rooms'].apply(lambda x: x + 'атная' if x not in ['Студия', 'Неизвестно'] else x)
    # Оставляем только квартиры дешевле константы MAX_PRICE
    df = df[df['price'] < 45000000]
    # Избавляемся от дубликатов
    df.drop_duplicates(inplace=True)

    df.to_csv(output_path, index=False)

if __name__ == '__main__':
    clean_data()