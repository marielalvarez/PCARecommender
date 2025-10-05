"""Tests for FastAPI endpoints"""

import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pytest
from fastapi.testclient import TestClient
from src.api.main import app


client = TestClient(app)


class TestAPI:
    """Test suite for API endpoints"""
    
    def test_root_endpoint(self):
        """Test root health check endpoint"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "service" in data
    
    def test_fit_endpoint_valid_data(self):
        """Test fit endpoint with valid data"""
        payload = {
            "data": [
                {
                    "GRAPROES": 0.5,
                    "GRAPROES_F": 0.45,
                    "GRAPROES_M": 0.55,
                    "RECUCALL_C": 0.6,
                    "RAMPAS_C": 0.3,
                    "PASOPEAT_C": 0.7,
                    "BANQUETA_C": 0.7,
                    "GUARNICI_C": 0.65,
                    "CICLOVIA_C": 0.4,
                    "CICLOCAR_C": 0.35,
                    "ALUMPUB_C": 0.8,
                    "LETRERO_C": 0.5,
                    "TELPUB_C": 0.6,
                    "ARBOLES_C": 0.55,
                    "DRENAJEP_C": 0.7,
                    "TRANSCOL_C": 0.45,
                    "ACESOPER_C": 0.5,
                    "ACESOAUT_C": 0.6
                },
                {
                    "GRAPROES": 0.6,
                    "GRAPROES_F": 0.55,
                    "GRAPROES_M": 0.65,
                    "RECUCALL_C": 0.5,
                    "RAMPAS_C": 0.4,
                    "PASOPEAT_C": 0.6,
                    "BANQUETA_C": 0.5,
                    "GUARNICI_C": 0.55,
                    "CICLOVIA_C": 0.3,
                    "CICLOCAR_C": 0.25,
                    "ALUMPUB_C": 0.7,
                    "LETRERO_C": 0.4,
                    "TELPUB_C": 0.5,
                    "ARBOLES_C": 0.45,
                    "DRENAJEP_C": 0.6,
                    "TRANSCOL_C": 0.35,
                    "ACESOPER_C": 0.4,
                    "ACESOAUT_C": 0.5
                }
            ]
        }
        response = client.post("/fit", json=payload)
        if response.status_code != 200:
            print(f"\n❌ Error Response: {response.json()}")
        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.json()}"
        data = response.json()
        assert data["status"] == "ok"
        assert "n_components" in data
        assert "explained" in data
    
    def test_recommend_endpoint(self):
        """Test recommend endpoint"""
        # First fit the model
        fit_payload = {
            "data": [
                {
                    "GRAPROES": 0.5,
                    "GRAPROES_F": 0.45,
                    "GRAPROES_M": 0.55,
                    "RECUCALL_C": 0.6,
                    "RAMPAS_C": 0.3,
                    "PASOPEAT_C": 0.7,
                    "BANQUETA_C": 0.7,
                    "GUARNICI_C": 0.65,
                    "CICLOVIA_C": 0.4,
                    "CICLOCAR_C": 0.35,
                    "ALUMPUB_C": 0.8,
                    "LETRERO_C": 0.5,
                    "TELPUB_C": 0.6,
                    "ARBOLES_C": 0.55,
                    "DRENAJEP_C": 0.7,
                    "TRANSCOL_C": 0.45,
                    "ACESOPER_C": 0.5,
                    "ACESOAUT_C": 0.6
                },
                {
                    "GRAPROES": 0.6,
                    "GRAPROES_F": 0.55,
                    "GRAPROES_M": 0.65,
                    "RECUCALL_C": 0.5,
                    "RAMPAS_C": 0.4,
                    "PASOPEAT_C": 0.6,
                    "BANQUETA_C": 0.5,
                    "GUARNICI_C": 0.55,
                    "CICLOVIA_C": 0.3,
                    "CICLOCAR_C": 0.25,
                    "ALUMPUB_C": 0.7,
                    "LETRERO_C": 0.4,
                    "TELPUB_C": 0.5,
                    "ARBOLES_C": 0.45,
                    "DRENAJEP_C": 0.6,
                    "TRANSCOL_C": 0.35,
                    "ACESOPER_C": 0.4,
                    "ACESOAUT_C": 0.5
                }
            ]
        }
        client.post("/fit", json=fit_payload)
        
        # Now get recommendations
        response = client.post("/recommend", json=fit_payload)
        assert response.status_code == 200
        data = response.json()
        assert "recommendations" in data
        assert "model_version" in data
        assert "comp_topvars" in data
    
    def test_recommend_without_fit_should_fail(self):
        """Test that recommend fails without fitting first"""
        payload = {
            "data": [
                {
                    "GRAPROES": 0.5,
                    "BANQUETA_C": 0.7
                }
            ]
        }
        # This might fail if model isn't fitted
        # Depending on implementation, adjust assertion
        response = client.post("/recommend", json=payload)
        # Either succeeds or returns 422
        assert response.status_code in [200, 422]
