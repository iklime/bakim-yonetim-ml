from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="Bakim Yonetim ML Servisi")

# Egitilmis modeli ve encoder\u0027i yukle
model = joblib.load("failure_model.pkl")
type_encoder = joblib.load("type_encoder.pkl")

class MakineVerisi(BaseModel):
    tip: str  # "L", "M" veya "H"
    hava_sicakligi: float
    islem_sicakligi: float
    donus_hizi: float
    tork: float
    alet_asinmasi: float

@app.get("/")
def anasayfa():
    return {"mesaj": "Bakim Yonetim ML Servisi calisiyor"}

@app.post("/tahmin")
def tahmin_yap(veri: MakineVerisi):
    tip_encoded = type_encoder.transform([veri.tip])[0]

    ozellikler = pd.DataFrame([{
        "Type_encoded": tip_encoded,
        "Air temperature": veri.hava_sicakligi,
        "Process temperature": veri.islem_sicakligi,
        "Rotational speed": veri.donus_hizi,
        "Torque": veri.tork,
        "Tool wear": veri.alet_asinmasi,
    }])

    olasilik = model.predict_proba(ozellikler)[0][1]
    tahmin = model.predict(ozellikler)[0]

    return {
        "ariza_olasiligi": round(float(olasilik), 4),
        "ariza_tahmini": bool(tahmin),
        "guven_skoru": round(float(max(model.predict_proba(ozellikler)[0])), 4)
    }
