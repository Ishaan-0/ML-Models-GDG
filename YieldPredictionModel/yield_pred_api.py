from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()

class InputData(BaseModel):
    Dist_Name: str  # Update with actual data types
    Crop: str
    Area: float
    Production: float

@app.post("/predict/")
def predict(data: InputData):
    from model import get_input, prediction
    input_dataframe = get_input(data)
    prediction = prediction(input_dataframe)
    return {"predicted_yield": prediction}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
