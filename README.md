# PCA Recommender

Sistema de recomendaciones de infraestructura urbana basado en Análisis de Componentes Principales (PCA).

## 📋 Descripción

Este proyecto implementa un sistema de recomendaciones que analiza métricas de infraestructura urbana de diferentes zonas y sugiere intervenciones prioritarias utilizando PCA. El sistema identifica el componente más débil de cada zona y recomienda mejoras específicas.

## 🏗️ Estructura del Proyecto

```
PCARecommender/
├── src/
│   ├── api/
│   │   ├── __init__.py
│   │   ├── main.py          # FastAPI application
│   │   └── schemas.py       # Pydantic models
│   ├── models/
│   │   ├── __init__.py
│   │   └── pca_recommender.py  # Modelo principal de PCA
│   └── __init__.py
├── notebooks/
│   └── pruebas_pcarecommender.ipynb  # Notebooks de análisis
├── tests/
│   └── (tests here)
├── data/
│   └── (data files here)
├── requirements.txt
├── README.md
└── .gitignore
```

## 🚀 Instalación

1. Clona el repositorio:
```bash
git clone https://github.com/marielalvarez/PCARecommender.git
cd PCARecommender
```

2. Crea un entorno virtual:
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

3. Instala las dependencias:
```bash
pip install -r requirements.txt
```

## 📦 Uso

### Como API

Inicia el servidor FastAPI:

```bash
cd src/api
uvicorn main:app --reload
```

La API estará disponible en `http://localhost:8000`

Endpoints disponibles:
- `GET /` - Health check
- `POST /fit` - Entrena el modelo con datos
- `POST /recommend` - Genera recomendaciones

### Como Librería Python

```python
from src.models import PCARecommender
import pandas as pd

# Crear el modelo
recommender = PCARecommender(var_target=0.80, top_k_loadings=5)

# Entrenar
recommender.fit(df)

# Generar recomendaciones
results = recommender.transform(df)
print(results['recommendations'])
```

## 🔧 Configuración

El modelo acepta las siguientes columnas de datos:

- `GRAPROES`, `GRAPROES_F`, `GRAPROES_M`: Escuelas
- `BANQUETA_C`: Banquetas
- `RAMPAS_C`: Rampas de accesibilidad
- `PASOPEAT_C`: Pasos peatonales
- `CICLOVIA_C`: Ciclovías
- `ALUMPUB_C`: Alumbrado público
- `ARBOLES_C`: Arbolado urbano
- Y más...

## 📊 Notebooks

Explora los análisis en el directorio `notebooks/`:
- `pruebas_pcarecommender.ipynb`: Pruebas y ejemplos de uso

## 🧪 Tests

Ejecuta las pruebas:

```bash
pytest tests/
```

## 📝 Licencia

[Tu licencia aquí]

## 👥 Autores

- Mariel Alvarez (@marielalvarez)

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Por favor, abre un issue primero para discutir los cambios que te gustaría hacer.
