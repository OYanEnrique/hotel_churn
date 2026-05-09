from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
from pathlib import Path
import io
import requests


app = FastAPI()


BASE_DIR = Path(__file__).resolve().parents[1]
PREP_PATH = BASE_DIR / "models" / "preprocessor.joblib"

# Modelo carregado diretamente do Google Drive em memória (sem salvar em disco)
MODEL_DRIVE_URL = "https://drive.google.com/uc?export=download&id=18KwDP5GL_aw9LNV5HkBQ3I8k40NwiQ8d"

if PREP_PATH.exists():
    response = requests.get(MODEL_DRIVE_URL, timeout=60)
    response.raise_for_status()
    modelo = joblib.load(io.BytesIO(response.content))
    preprocessor = joblib.load(PREP_PATH)
else:
    modelo = None
    preprocessor = None


class ReservaHotel(BaseModel):
    lead_time: int
    adr: float
    total_pessoas: int
    total_noites: int
    mudou_quarto: int
    previous_cancellations: int
    hotel: str
    market_segment: str
    deposit_type: str
    customer_type: str
    tem_filhos: str


@app.post("/prever_churn")
def prever(cliente: ReservaHotel):
    if modelo is None or preprocessor is None:
        raise HTTPException(status_code=500, detail="Modelo ou preprocessor não carregado no servidor")
    cliente_dict = cliente.model_dump()
    cliente_df = pd.DataFrame([cliente_dict])
    try:
        cliente_tratado = preprocessor.transform(cliente_df)
        predicao = modelo.predict(cliente_tratado)
        return {"churn_predito": int(predicao[0])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
