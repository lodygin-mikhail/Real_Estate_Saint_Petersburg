from enum import Enum

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field
from pydantic.v1 import PositiveInt, PositiveFloat

app = FastAPI()

model = joblib.load("models/real_estate_pipeline.joblib")

class PropertyTypeEnum(str, Enum):
    STUDIO = "Студия"
    ONE_ROOM = "1-комнатная"
    TWO_ROOMS = "южный"
    THREE_ROOMS = "восточный"
    FOUR_ROOMS = "западный"
    FIVE_ROOMS = ''

class PropertyFeatures(BaseModel):
    flat_status: bool = Field(description='Сдаётся ли в данный момент квартира?')
    num_of_rooms: str
    total_area: PositiveFloat
    living_area: PositiveFloat
    kitchen_area: PositiveFloat
    floor: PositiveInt
    metro_station: str
    minutes_to_metro: PositiveInt
    transfer_type: str
    house_age: PositiveInt
    is_future_building: bool

class ModelPrediction(BaseModel):
    price: float

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(features: PropertyFeatures):
    input_data = {
        'flat_status': [features.flat_status],
        'num_of_rooms': [features.num_of_rooms],
        'total_area_m2': [features.total_area],
        'living_area_m2': [features.living_area],
        'kitchen_area_m2': [features.kitchen_area],
        'floor': [features.floor],
        'metro_station': [features.metro_station],
        'minutes_to_metro': [features.minutes_to_metro],
        'transfer_type': [features.transfer_type],
        'house_age': [features.house_age],
        'is_future_building': [features.is_future_building]
    }
    data = pd.DataFrame(input_data)

    prediction = model.predict(data)

    return {"prediction": ModelPrediction(price=prediction)}


if __name__ == '__main__':
    import uvicorn
    uvicorn.run('main:app', reload=True)