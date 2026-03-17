from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# Load trained model
model = joblib.load("models/churn_model.pkl")

# Define request schema
class Features(BaseModel):
    features: list[float]

@app.get("/")
def home():
    return {"message": "Customer Churn Prediction API"}

@app.post("/predict")
def predict(data: Features):
    arr = np.array(data.features).reshape(1, -1)
    prediction = model.predict(arr)
    return {"churn_prediction": int(prediction[0])}