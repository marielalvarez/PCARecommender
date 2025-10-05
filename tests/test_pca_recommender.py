"""Unit tests for PCARecommender model"""

import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pytest
import pandas as pd
import numpy as np
from src.models import PCARecommender, DEFAULT_BASE_COLS


class TestPCARecommender:
    """Test suite for PCARecommender class"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing"""
        np.random.seed(42)
        data = {col: np.random.rand(10) for col in DEFAULT_BASE_COLS}
        return pd.DataFrame(data)
    
    @pytest.fixture
    def recommender(self):
        """Create a PCARecommender instance"""
        return PCARecommender(var_target=0.80, top_k_loadings=5)
    
    def test_initialization(self, recommender):
        """Test that recommender initializes correctly"""
        assert recommender.var_target == 0.80
        assert recommender.top_k_loadings == 5
        assert recommender.pca is None
        assert recommender.cols == DEFAULT_BASE_COLS
    
    def test_fit(self, recommender, sample_data):
        """Test fitting the model"""
        recommender.fit(sample_data)
        
        assert recommender.pca is not None
        assert recommender.scaler is not None
        assert recommender.imputer is not None
        assert len(recommender.cols_used_) > 0
        assert recommender.explained_ is not None
    
    def test_transform(self, recommender, sample_data):
        """Test transforming data after fitting"""
        recommender.fit(sample_data)
        results = recommender.transform(sample_data)
        
        assert "recommendations" in results
        assert "scores" in results
        assert "loadings" in results
        assert "explained" in results
        assert len(results["recommendations"]) == len(sample_data)
    
    def test_fit_transform(self, recommender, sample_data):
        """Test fit_transform method"""
        results = recommender.fit_transform(sample_data)
        
        assert "recommendations" in results
        assert len(results["recommendations"]) == len(sample_data)
    
    def test_transform_without_fit_raises_error(self, recommender, sample_data):
        """Test that transform without fit raises an error"""
        with pytest.raises(RuntimeError):
            recommender.transform(sample_data)
    
    def test_save_load(self, recommender, sample_data, tmp_path):
        """Test saving and loading the model"""
        # Fit and save
        recommender.fit(sample_data)
        model_path = tmp_path / "model.joblib"
        recommender.save(str(model_path))
        
        # Load
        loaded_recommender = PCARecommender.load(str(model_path))
        
        # Test that loaded model works
        results = loaded_recommender.transform(sample_data)
        assert "recommendations" in results
        assert len(results["recommendations"]) == len(sample_data)
