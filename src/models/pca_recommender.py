# pca_recommender.py
from __future__ import annotations
import numpy as np, pandas as pd, joblib, warnings
from typing import Dict, List, Optional, Any, Tuple
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

DEFAULT_BASE_COLS = [
    'GRAPROES','GRAPROES_F','GRAPROES_M','RECUCALL_C','RAMPAS_C','PASOPEAT_C',
    'BANQUETA_C','GUARNICI_C','CICLOVIA_C','CICLOCAR_C','ALUMPUB_C','LETRERO_C',
    'TELPUB_C','ARBOLES_C','DRENAJEP_C','TRANSCOL_C','ACESOPER_C','ACESOAUT_C'
]

DEFAULT_INTERV_MAP = {
    'GRAPROES': 'Escuelas','GRAPROES_F': 'Escuelas','GRAPROES_M': 'Escuelas',
    'ARBOLES_C':'Urban forestry (arbolado / pocket parks / sombreaderos)',
    'BANQUETA_C':'Sidewalks & walkability (banquetas)',
    'PASOPEAT_C':'Crosswalks / cruces peatonales',
    'RAMPAS_C':'Accessibility (rampas PMR)',
    'ALUMPUB_C':'Public lighting (alumbrado)',
    'CICLOVIA_C':'Bike lanes (ciclovía)',
    'CICLOCAR_C':'Traffic calming / carriles bici-seguro',
    'DRENAJEP_C':'Stormwater / drenaje pluvial',
    'RECUCALL_C':'Street maintenance / recarpeteo',
    'GUARNICI_C':'Curbs / guarniciones',
    'LETRERO_C':'Wayfinding & signage',
    'TELPUB_C':'Street furniture / telecom',
    'ACESOPER_C':'Senderos operativos / servidumbres',
    'ACESOAUT_C':'Gestión de acceso vehicular',

}

class PCARecommender:
    """
    Recomienda intervención por zona a partir de PCA:
    - Selecciona componente más "débil" (score mínimo por fila)
    - Dentro de ese componente, toma la variable con peor z-score y la mapea a intervención
    """
    def __init__(self,
                 cols: Optional[List[str]] = None,
                 interv_map: Optional[Dict[str,str]] = None,
                 var_target: float = 0.80,
                 top_k_loadings: int = 5,
                 model_version: str = "v1.0"):
        self.cols = cols or DEFAULT_BASE_COLS
        self.interv_map = interv_map or DEFAULT_INTERV_MAP
        self.var_target = float(var_target)
        self.top_k_loadings = int(top_k_loadings)
        self.model_version = model_version

        # artefactos sklearn
        self.imputer: Optional[SimpleImputer] = None
        self.scaler: Optional[StandardScaler] = None
        self.pca: Optional[PCA] = None

        # metadatos
        self.cols_used_: List[str] = []
        self.comp_topvars_: Dict[str, List[str]] = {}
        self.loadings_: Optional[pd.DataFrame] = None
        self.explained_: Optional[pd.DataFrame] = None

    # ---------- helpers ----------
    @staticmethod
    def _ensure_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        out = df.copy()
        for c in cols:
            out[c] = pd.to_numeric(out[c], errors="coerce")
        return out

    def _pick_components(self, pca_full: PCA) -> int:
        cum = np.cumsum(pca_full.explained_variance_ratio_)
        return int(np.searchsorted(cum, self.var_target) + 1)

    # ---------- core ----------
    def fit(self, df: pd.DataFrame) -> "PCARecommender":
        # valida columnas presentes
        cols_use = [c for c in self.cols if c in df.columns]
        if not cols_use:
            raise ValueError("Ninguna de las columnas esperadas está en el DataFrame.")
        self.cols_used_ = cols_use

        X = self._ensure_numeric(df[self.cols_used_], self.cols_used_).values

        # imputar + escalar
        self.imputer = SimpleImputer(strategy="median")
        X_imp = self.imputer.fit_transform(X)

        self.scaler = StandardScaler()
        Xz = self.scaler.fit_transform(X_imp)

        # PCA full para decidir n_components
        pca_full = PCA(svd_solver="full")  # determinista
        pca_full.fit(Xz)
        n_comp = self._pick_components(pca_full)

        # PCA final
        self.pca = PCA(n_components=n_comp, svd_solver="full")
        Z = self.pca.fit_transform(Xz)

        # loadings
        load = pd.DataFrame(self.pca.components_, columns=self.cols_used_)
        self.loadings_ = load.copy()
        la = load.abs()
        self.comp_topvars_ = {
            f"PC{k+1}": la.iloc[k].sort_values(ascending=False).head(self.top_k_loadings).index.tolist()
            for k in range(n_comp)
        }

        self.explained_ = pd.DataFrame({
            "component": [f"PC{k+1}" for k in range(n_comp)],
            "explained_variance_ratio": self.pca.explained_variance_ratio_,
            "cumulative_variance": np.cumsum(self.pca.explained_variance_ratio_)
        })
        return self

    def transform(self, df: pd.DataFrame) -> Dict[str, Any]:
        if any(obj is None for obj in [self.imputer, self.scaler, self.pca, self.cols_used_]):
            raise RuntimeError("Debes llamar fit() antes de transform().")

        # asegurar columnas (faltantes -> NaN)
        Xdf = df.reindex(columns=self.cols_used_, fill_value=np.nan)
        Xdf = self._ensure_numeric(Xdf, self.cols_used_)

        X_imp = self.imputer.transform(Xdf.values)
        Xz = self.scaler.transform(X_imp)
        Z = self.pca.transform(Xz)  # (n, k)

        n, k = Z.shape
        comp_names = np.array([f"PC{i+1}" for i in range(k)])
        # componente más débil por fila (score mínimo)
        weak_idx = np.argmin(Z, axis=1)              # (n,)
        weak_comp = comp_names[weak_idx]             # (n,)

        # peor variable dentro del componente "débil"
        # necesitamos mirar los z-scores de las top vars de ese comp por fila
        worst_feature = []
        worst_value = []
        for i in range(n):
            pc = weak_comp[i]
            vars_k = self.comp_topvars_.get(pc, self.cols_used_)  # fallback: todas
            # índices de columnas
            ix = [self.cols_used_.index(v) for v in vars_k if v in self.cols_used_]
            if not ix:
                worst_feature.append(None)
                worst_value.append(np.nan)
                continue
            row_vals = Xz[i, ix]
            j = int(np.argmin(row_vals))
            worst_feature.append(vars_k[j])
            worst_value.append(float(row_vals[j]))

        rec_interv = [self.interv_map.get(v, f"Mejorar '{v}'") if v is not None else "Sin recomendación"
                      for v in worst_feature]

        # construir DataFrame de salida
        out = pd.DataFrame({
            "weak_component": weak_comp,
            "weak_score": Z[np.arange(n), weak_idx],
            "worst_feature": worst_feature,
            "worst_feature_z": worst_value,
            "recommended_intervention": rec_interv
        })

        # empaquetar diagnósticos
        loadings_named = self.loadings_.copy()
        loadings_named.index = [f"PC{i+1}" for i in range(k)]
        explained = self.explained_.copy()

        return {
            "recommendations": out,
            "scores": pd.DataFrame(Z, columns=[f"PC{i+1}" for i in range(k)]),
            "loadings": loadings_named,
            "explained": explained,
            "comp_topvars": self.comp_topvars_,
            "model_version": self.model_version,
            "columns_used": self.cols_used_
        }

    def fit_transform(self, df: pd.DataFrame) -> Dict[str, Any]:
        return self.fit(df).transform(df)

    # ----- persistencia -----
    def save(self, path: str) -> None:
        payload = {
            "imputer": self.imputer,
            "scaler": self.scaler,
            "pca": self.pca,
            "cols_used_": self.cols_used_,
            "comp_topvars_": self.comp_topvars_,
            "loadings_": self.loadings_,
            "explained_": self.explained_,
            "interv_map": self.interv_map,
            "var_target": self.var_target,
            "top_k_loadings": self.top_k_loadings,
            "model_version": self.model_version,
        }
        joblib.dump(payload, path)

    @classmethod
    def load(cls, path: str) -> "PCARecommender":
        payload = joblib.load(path)
        obj = cls(
            cols=payload.get("cols_used_", DEFAULT_BASE_COLS),
            interv_map=payload.get("interv_map", DEFAULT_INTERV_MAP),
            var_target=payload.get("var_target", 0.80),
            top_k_loadings=payload.get("top_k_loadings", 5),
            model_version=payload.get("model_version", "v1.0")
        )
        obj.imputer = payload["imputer"]
        obj.scaler = payload["scaler"]
        obj.pca = payload["pca"]
        obj.cols_used_ = payload["cols_used_"]
        obj.comp_topvars_ = payload["comp_topvars_"]
        obj.loadings_ = payload["loadings_"]
        obj.explained_ = payload["explained_"]
        return obj
