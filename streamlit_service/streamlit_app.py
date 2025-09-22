import json
import os
from pathlib import Path
import requests
from requests.exceptions import ConnectionError
import streamlit as st

IMG_PATH = Path(__file__).parent / "img"

FASTAPI_URL = os.getenv("FASTAPI_URL", "http://fastapi-service:6000")

with open("unique_values.json", "r", encoding="utf-8") as f:
    unique_values = json.load(f)

st.title("Real Estate in Saint-Petersburg")

st.image(IMG_PATH / "buildings.avif")

st.sidebar.subheader("Выберете параметры квартиры")

station = st.sidebar.selectbox(
    "Выберете станцию метро",
    unique_values["metro_station"],
    key="metro_station",
)

flat_type = st.sidebar.selectbox(
    "Выберете тип квартиры",
    unique_values["num_of_rooms"],
    key="flat_type",
)

total_area = st.sidebar.slider(
    "Укажите площадь квартиры в м\u00b2",
    15.0,
    150.0,
    30.0,
    format="%.1f",
    key="total_area",
)

living_area = st.sidebar.slider(
    "Укажите жилую площадь квартиры в м\u00b2",
    15.0,
    150.0,
    25.0,
    format="%.1f",
    key="living_area",
)

kitchen_area = st.sidebar.slider(
    "Укажите площадь кухни в м\u00b2",
    0.0,
    25.0,
    10.0,
    format="%.1f",
    key="kitchen_area",
)

floor = st.sidebar.slider("Укажите этаж", 1, 30, 1, key="floor")

with st.sidebar:
    left, right = st.columns(2, vertical_alignment="bottom")

    minutes_to_metro = left.slider(
        "Транспортная доступность в мин", 1, 60, 10, key="minutes_to_metro"
    )

    transport_type = right.selectbox(
        "Вид транспорта",
        ["пешком", "на машине"],
        label_visibility="hidden",
        key="transport_type",
    )

building_age = st.sidebar.slider("Укажите возраст здания", 0, 150, key="building_age")

flat_status = st.sidebar.toggle("Квартира сейчас сдаётся", key="flat_status")

is_future_building = st.sidebar.toggle(
    "Квартира еще строится", key="is_future_building"
)

if st.sidebar.button("Предсказать цену"):
    data = {
        "flat_status": int(flat_status),
        "num_of_rooms": flat_type,
        "total_area": total_area,
        "living_area": living_area,
        "kitchen_area": kitchen_area,
        "floor": floor,
        "metro_station": station,
        "minutes_to_metro": minutes_to_metro,
        "transfer_type": transport_type,
        "house_age": building_age,
        "is_future_building": int(is_future_building),
    }
    try:
        response = requests.post(f"{FASTAPI_URL}/predict", json=data)
        if response.status_code == 200:
            prediction = response.json()["prediction"]["price"]
            st.success(f"Стоимость недвижимости: {prediction:.2f} рублей")
        else:
            st.error(f"Request failed with status code {response.status_code}")
    except ConnectionError as e:
        st.error(f"Failed to connect to the server")
