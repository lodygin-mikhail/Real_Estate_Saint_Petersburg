from pathlib import Path
import requests
from requests.exceptions import ConnectionError
import joblib
import streamlit as st
import pandas as pd
import numpy as np

ROOT_DIR = Path(__file__).parent.parent
IMG_PATH = ROOT_DIR / 'img'
MODEL_PATH = ROOT_DIR / 'model'

ip_api = "127.0.0.1"
port_api = "500"

@st.cache_data
def load_flat_type(path:Path):
    flat_types = joblib.load(path / 'flat_type_names.joblib')
    flat_types.sort()
    return np.delete(flat_types, 5)

@st.cache_data
def load_metro_stations(path:Path):
    stations = joblib.load(path / 'metro_station_names.joblib')
    stations.sort()
    return stations

st.title('Real Estate in Saint-Petersburg')

st.image(IMG_PATH / 'buildings.avif')

st.sidebar.subheader('Выберете параметры квартиры')

station = st.sidebar.selectbox('Выберете станцию метро', load_metro_stations(MODEL_PATH), key='metro_station')

flat_type = st.sidebar.selectbox('Выберете тип квартиры', load_flat_type(MODEL_PATH), key='flat_type')

total_area = st.sidebar.slider('Укажите площадь квартиры в м\u00b2', 0.0, 250.0, 30.0, format='%.1f', key = 'total_area')

living_area = st.sidebar.slider('Укажите жилую площадь квартиры в м\u00b2', 0.0, 200.0, 25.0, format='%.1f', key = 'living_area')

kitchen_area = st.sidebar.slider('Укажите площадь кухни в м\u00b2', 0.0, 50.0, 10.0, format='%.1f', key = 'kitchen_area')

floor = st.sidebar.slider('Укажите этаж', 1, 50, 1, key = 'floor')

with st.sidebar:
    left, right = st.columns(2, vertical_alignment="bottom")

    minutes_to_metro = left.slider("Транспортная доступность в мин", 1, 60, 1, key = 'minutes_to_metro')

    transport_type = right.selectbox("", ["пешком", "на машине"], key = 'transport_type')

building_age = st.sidebar.slider('Укажите возраст здания', 0, 150, key = 'building_age')

flat_status = st.sidebar.toggle('Квартира сейчас сдаётся', key = 'flat_status')

is_future_building = st.sidebar.toggle('Квартира еще строится', key = 'is_future_building')

