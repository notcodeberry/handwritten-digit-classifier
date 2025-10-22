from typing import Union
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from backend.model.load_model import ClassificationModel

app = FastAPI()

origins = [
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)
class DigitFeatures(BaseModel):
    encodedImage: str
class PredictionResult(BaseModel):
    prediction: int
    confidence: float
    message: str

model = ClassificationModel("model/classification_model.pth")

@app.post("/predict", response_model = PredictionResult)
def predict_digit(features: DigitFeatures):
    prediction, confidence = model.predict(features.encodedImage)

    return {"prediction": prediction, "confidence": confidence, "message": "ok"}