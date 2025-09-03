import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

model = joblib.load("models/real_estate_pipeline.joblib")

class PropertyFeatures(BaseModel):
    flat_status: bool
    num_of_rooms: str
    total_area: float
    living_area: float
    kitchen_area: float
    floor: int
    metro_station: str
    minutes_to_metro: int
    transfer_type: str
    house_age: int
    is_future_building: bool

class ModelPrediction(BaseModel):
    price: float

@app.post("/predict")
async def predict(data: PropertyFeatures):
     input_data = {
        'flat_status': [PropertyFeatures.flat_status],
        'num_of_rooms': [PropertyFeatures.num_of_rooms],
        'total_area_m2': [PropertyFeatures.total_area],
        'living_area_m2': [PropertyFeatures.living_area],
        'kitchen_area_m2': [PropertyFeatures.kitchen_area],
        'floor': [PropertyFeatures.floor],
        'metro_station': [PropertyFeatures.metro_station],
        'minutes_to_metro': [PropertyFeatures.minutes_to_metro],
        'transfer_type': [PropertyFeatures.transfer_type],
        'house_age': [PropertyFeatures.house_age],
        'is_future_building': [PropertyFeatures.is_future_building]
     }
    prediction = model.predict(input_data)
    return ModelPrediction(price=prediction)


print(model.predict(X))

# if __name__ == '__main__':
#     import uvicorn
#     uvicorn.run('main:app', reload=True)