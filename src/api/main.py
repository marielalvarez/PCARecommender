"""FastAPI application for Urban PCA Recommender"""

from fastapi import HTTPException, Query
import pandas as pd
from enum import Enum

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

class Action(str, Enum):
    fit = "fit"
    recommend = "recommend"
    fit_and_recommend = "fit_and_recommend"

@app.post(
    "/pca",
    summary="Entrena el PCA y/o genera recomendaciones según 'action'"
)
def pca(payload: Payload, action: Action = Query(default=Action.fit_and_recommend)):
    """
    Usa el mismo payload para entrenar el modelo PCA y/o generar recomendaciones.
    action:
      - 'fit'               -> solo entrena
      - 'recommend'         -> solo recomienda (requiere modelo ya entrenado)
      - 'fit_and_recommend' -> entrena y recomienda en la misma llamada
    """
    df = pd.DataFrame([r.model_dump() for r in payload.data])

    try:
        response = {"status": "ok"}

        # 1) Entrenamiento
        if action in (Action.fit, Action.fit_and_recommend):
            recommender.fit(df)
            response.update({
                "fit": {
                    "n_components": len(recommender.pca.components_),
                    "explained": recommender.explained_.to_dict(orient="records"),
                    "columns_used": recommender.cols_used_,
                }
            })

        # 2) Recomendaciones
        if action in (Action.recommend, Action.fit_and_recommend):
            res = recommender.transform(df)
            response.update({
                "recommend": {
                    "model_version": res["model_version"],
                    "columns_used": res["columns_used"],
                    "explained": res["explained"].to_dict(orient="records"),
                    "comp_topvars": res["comp_topvars"],
                    "recommendations": res["recommendations"].to_dict(orient="records"),
                }
            })

        return response

    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))