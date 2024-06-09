# from fastapi import FastAPI, File, UploadFile
# from pydantic import BaseModel
# from model.model import * 
# import numpy as np
# import json

# app = FastAPI()

# class InputUser(BaseModel):
#     value: np.ndarray

# class PredictionOut(BaseModel):
#     prediction: str

# @app.get("/")
# def home():
#     model_version = "1.0.0"
#     return {"health_check": "OK", "model_version": model_version}

# @app.post("/predict", response_model=PredictionOut)
# def predict(input_audio):
#     # Membaca file WAV dan mengonversinya ke dalam array NumP
#     prediction = predict_tabular(input_audio)
#     return json.dumps({"prediction": prediction})