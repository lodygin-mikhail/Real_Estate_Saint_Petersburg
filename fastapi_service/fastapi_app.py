from enum import Enum
from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI()

model = joblib.load("model.joblib")


class PropertyTypeEnum(str, Enum):
    STUDIO = "Студия"
    ONE_ROOM = "1-комнатная"
    TWO_ROOMS = "2-комнатная"
    THREE_ROOMS = "3-комнатная"
    FOUR_ROOMS = "4-комнатная"
    FIVE_ROOMS = "5-комнатная"


class MetroGetTypeEnum(str, Enum):
    LEGS = "пешком"
    CAR = "на машине"


class PropertyFeatures(BaseModel):
    flat_status: bool = Field(description="Сдаётся ли в данный момент квартира")
    num_of_rooms: PropertyTypeEnum = Field(description="Количество комнат")
    total_area: float = Field(gt=0, description="Общая площадь квартиры")
    living_area: float = Field(gt=0, description="Жилая площадь квартиры")
    kitchen_area: float = Field(gt=0, description="Площадь кухни")
    floor: int = Field(gt=0, description="Этаж квартиры")
    metro_station: str = Field(description="Ближайшая станция метро")
    minutes_to_metro: int = Field(gt=0, description="Количество минут до метро")
    transfer_type: MetroGetTypeEnum = Field(description="Способ добраться до метро")
    house_age: int = Field(ge=0, description="Сколько лет прошло с постройки дома")
    is_future_building: bool = Field(description="Находится ли дом в этапе стройки")


class ModelPrediction(BaseModel):
    price: float = Field(gt=0)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(features: PropertyFeatures):
    input_data = {
        "flat_status": [features.flat_status],
        "num_of_rooms": [features.num_of_rooms],
        "total_area_m2": [features.total_area],
        "living_area_m2": [features.living_area],
        "kitchen_area_m2": [features.kitchen_area],
        "floor": [features.floor],
        "metro_station": [features.metro_station],
        "minutes_to_metro": [features.minutes_to_metro],
        "transfer_type": [features.transfer_type],
        "house_age": [features.house_age],
        "is_future_building": [features.is_future_building],
    }
    data = pd.DataFrame(input_data)

    prediction = model.predict(data)

    return {"prediction": ModelPrediction(price=prediction)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("fastapi_app:app", reload=True, host="0.0.0.0", port=5000)
