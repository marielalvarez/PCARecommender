# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import pandas as pd

from pca_recommender import PCARecommender, DEFAULT_BASE_COLS

app = FastAPI(title="Urban PCA Recommender", version="1.0")

# schemas
class Record(BaseModel):
    GRAPROES: Optional[float] = None
    GRAPROES_F: Optional[float] = None
    GRAPROES_M: Optional[float] = None
    RECUCALL_C: Optional[float] = None
    RAMPAS_C: Optional[float] = None
    PASOPEAT_C: Optional[float] = None
    BANQUETA_C: Optional[float] = None
    GUARNICI_C: Optional[float] = None
    CICLOVIA_C: Optional[float] = None
    CICLOCAR_C: Optional[float] = None
    ALUMPUB_C: Optional[float] = None
    LETRERO_C: Optional[float] = None
    TELPUB_C: Optional[float] = None
    ARBOLES_C: Optional[float] = None
    DRENAJEP_C: Optional[float] = None
    TRANSCOL_C: Optional[float] = None
    ACESOPER_C: Optional[float] = None
    ACESOAUT_C: Optional[float] = None
    PUESSEMI_C: Optional[float] = None
    PUESAMBU_C: Optional[float] = None

class Payload(BaseModel):
    data: List[Record] = Field(..., description="Lista de zonas con métricas")

recommender = PCARecommender(cols=DEFAULT_BASE_COLS, var_target=0.80, top_k_loadings=5)

@app.post("/fit", summary="Entrena el PCA con los datos enviados")
def fit(payload: Payload):
    df = pd.DataFrame([r.dict() for r in payload.data])
    try:
        recommender.fit(df)
        return {
            "status": "ok",
            "n_components": len(recommender.pca.components_),
            "explained": recommender.explained_.to_dict(orient="records"),
            "columns_used": recommender.cols_used_
        }
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))

@app.post("/recommend", summary="Genera recomendaciones por zona")
def recommend(payload: Payload):
    df = pd.DataFrame([r.dict() for r in payload.data])
    try:
        res = recommender.transform(df)
        out = {
            "model_version": res["model_version"],
            "columns_used": res["columns_used"],
            "explained": res["explained"].to_dict(orient="records"),
            "comp_topvars": res["comp_topvars"],
            "recommendations": res["recommendations"].to_dict(orient="records"),
        }
        return out
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))
