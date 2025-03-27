from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()


class Item(BaseModel):
    N: float
    P: float
    K: float
    temperature: float
    humidity: float
    ph: float
    rainfall: float
# Load the trained model and encoder
model = joblib.load("crop_recommendation_model.pkl")
encoder = joblib.load("label_encoder.pkl")

@app.post("/predict/")
def predict_crop(features: Item):
    N = Item.N
    P = Item.P
    K = Item.K 
    temperature = Item.temperature
    humidity = Item.humidity
    ph = Item.ph
    rainfall = Item.rainfall
    input_data = [N, P, K, temperature, humidity, ph, rainfall]
    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)[0]
    predicted_crop = encoder.inverse_transform([prediction])[0]
    return {"recommended_crop": predicted_crop}
