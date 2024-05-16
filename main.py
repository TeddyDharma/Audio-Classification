from fastapi import FastAPI, File, UploadFile
from model import * 
import numpy as np
import librosa
import io

app = FastAPI()

class PredictionOut(BaseModel):
    prediction: str

@app.get("/")
def home():
    model_version = "1.0.0"
    return {"health_check": "OK", "model_version": model_version}

@app.post("/predict", response_model=PredictionOut)
async def predict(file: UploadFile = File(...)):
    # Membaca file WAV dan mengonversinya ke dalam array NumPy
    contents = await file.read()
    audio, sr = librosa.load(io.BytesIO(contents), sr=None)
    
    # Menjalankan model untuk melakukan prediksi
    prediction = predict_tabular(audio)

    return {"prediction": prediction}