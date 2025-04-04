from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from main import  recommend_crop

app = FastAPI()

class Item(BaseModel):
    N: float
    P: float
    K: float
    temperature: float
    humidity: float
    ph: float
    rainfall: float

@app.post("/crop_predict/")
def predict_crop(features: Item):
    input_data = pd.DataFrame([{
        "N": features.N,
        "P": features.P,
        "K": features.K,
        "temperature": features.temperature,
        "humidity": features.humidity,
        "ph": features.ph,
        "rainfall": features.rainfall
    }])
    predicted_crop = recommend_crop(input_data)
    return {"recommended_crop": predicted_crop}
