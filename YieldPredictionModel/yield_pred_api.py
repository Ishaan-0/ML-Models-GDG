from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from model import yield_prediction, encoders_mapping

app = FastAPI()

class InputData(BaseModel):
    Dist_Name: str  # Update with actual data types
    Crop: str
    Area: float
    Production: float

@app.post("/yield_predict/")
def predict(data: InputData):
    dist_name, crop_name = encoders_mapping(data.Dist_Name.title(), data.Crop.upper())
    input_data = pd.DataFrame([{
        'Dist Name': dist_name,
        'Crop': crop_name,
        'Area': data.Area,
        'Production': data.Production
    }])
    prediction = yield_prediction(input_data)
    return {"predicted_yield": prediction}

