from fastapi import FastAPI
from app.model.model import predict_tabular
from pydantic import BaseModel
import uvicorn

app = FastAPI()

class InputUser(BaseModel):
    value: list

class PredictionOut(BaseModel):
    prediction: str

@app.get("/")
def home():
    model_version = "1.0.0"
    return {"health_check": "OK", "model_version": model_version}

@app.post("/predict", response_model=PredictionOut)
def predict(payload: InputUser):
    prediction = predict_tabular(payload.value)
    return {"prediction": prediction}



if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)