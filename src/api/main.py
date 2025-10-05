"""FastAPI application for Urban PCA Recommender"""

from fastapi import FastAPI, HTTPException
import pandas as pd

from .schemas import Payload
from ..models import PCARecommender, DEFAULT_BASE_COLS

app = FastAPI(
    title="Urban PCA Recommender",
    version="1.0",
    description="API for urban infrastructure recommendations using PCA analysis"
)

# Initialize the recommender model
recommender = PCARecommender(cols=DEFAULT_BASE_COLS, var_target=0.80, top_k_loadings=5)


@app.get("/", summary="Health check")
def root():
    """Check if the API is running"""
    return {
        "status": "ok",
        "service": "Urban PCA Recommender",
        "version": "1.0"
    }


@app.post("/fit", summary="Entrena el PCA con los datos enviados")
def fit(payload: Payload):
    """
    Train the PCA model with the provided zone data.
    
    Args:
        payload: List of zone records with infrastructure metrics
        
    Returns:
        Training results including number of components and explained variance
    """
    df = pd.DataFrame([r.model_dump() for r in payload.data])
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
    """
    Generate infrastructure recommendations for each zone.
    
    Args:
        payload: List of zone records with infrastructure metrics
        
    Returns:
        Recommendations with weak components and suggested interventions
    """
    df = pd.DataFrame([r.model_dump() for r in payload.data])
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
