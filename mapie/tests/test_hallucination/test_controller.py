"""
Tests for the main HallucinationController.
"""

import numpy as np
import pytest
from mapie.hallucination.controller import HallucinationController
from mapie.hallucination.losses import WeightedHallucinationLoss
from mapie.hallucination.policies import AbstractionPolicy


class TestHallucinationController:
    """Test main hallucination controller functionality."""
    
    def test_initialization(self):
        """Test controller initialization."""
        loss_fn = WeightedHallucinationLoss()
        policy = AbstractionPolicy()
        
        controller = HallucinationController(
            loss_function=loss_fn,
            policy=policy,
            random_state=42
        )
        
        assert controller.loss_function is loss_fn
        assert controller.policy is policy
        assert not controller.is_fitted_
        
    def test_fit_basic(self):
        """Test basic fitting functionality."""
        loss_fn = WeightedHallucinationLoss()
        policy = AbstractionPolicy()
        controller = HallucinationController(loss_fn, policy, random_state=42)
        
        # Simple calibration data
        cal_queries = ["What is 2+2?", "What is 3+3?", "What is 4+4?"]
        cal_responses = ["4", "6", "8"]
        cal_truth = ["4", "6", "8"]
        cal_confidence = np.array([0.9, 0.8, 0.7])
        
        controller.fit(
            calibration_queries=cal_queries,
            calibration_responses=cal_responses,
            calibration_ground_truth=cal_truth,
            alpha=0.1,
            confidence_scores=cal_confidence
        )
        
        assert controller.is_fitted_
        assert controller.lambda_star_ is not None
        assert controller.alpha_ == 0.1
        
    def test_predict_before_fit_raises_error(self):
        """Test that prediction before fitting raises error."""
        loss_fn = WeightedHallucinationLoss()
        policy = AbstractionPolicy()
        controller = HallucinationController(loss_fn, policy)
        
        with pytest.raises(ValueError, match="must be fitted"):
            controller.predict("query", "response")
            
    def test_predict_after_fit(self):
        """Test prediction after fitting."""
        loss_fn = WeightedHallucinationLoss()
        policy = AbstractionPolicy()
        controller = HallucinationController(loss_fn, policy, random_state=42)
        
        # Fit with simple data
        cal_queries = ["Q1", "Q2", "Q3", "Q4", "Q5"]
        cal_responses = ["A1", "A2", "A3", "A4", "A5"]
        cal_truth = ["A1", "A2", "A3", "A4", "A5"]
        cal_confidence = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
        
        controller.fit(
            calibration_queries=cal_queries,
            calibration_responses=cal_responses,
            calibration_ground_truth=cal_truth,
            alpha=0.1,
            confidence_scores=cal_confidence
        )
        
        # Test prediction
        response = controller.predict(
            query="Test query",
            base_response="Test response", 
            confidence_score=0.9
        )
        
        assert isinstance(response, str)
        assert len(response) > 0
        
    def test_predict_batch(self):
        """Test batch prediction."""
        loss_fn = WeightedHallucinationLoss()
        policy = AbstractionPolicy()
        controller = HallucinationController(loss_fn, policy, random_state=42)
        
        # Fit with simple data
        cal_queries = ["Q1", "Q2", "Q3", "Q4", "Q5"] 
        cal_responses = ["A1", "A2", "A3", "A4", "A5"]
        cal_truth = ["A1", "A2", "A3", "A4", "A5"]
        cal_confidence = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
        
        controller.fit(
            calibration_queries=cal_queries,
            calibration_responses=cal_responses,
            calibration_ground_truth=cal_truth,
            alpha=0.1,
            confidence_scores=cal_confidence
        )
        
        # Test batch prediction
        test_queries = ["TQ1", "TQ2"]
        test_responses = ["TR1", "TR2"]
        test_confidence = np.array([0.8, 0.6])
        
        batch_responses = controller.predict_batch(
            queries=test_queries,
            base_responses=test_responses,
            confidence_scores=test_confidence
        )
        
        assert len(batch_responses) == 2
        assert all(isinstance(resp, str) for resp in batch_responses)
        
    def test_score_method(self):
        """Test scoring method."""
        loss_fn = WeightedHallucinationLoss()
        policy = AbstractionPolicy()
        controller = HallucinationController(loss_fn, policy, random_state=42)
        
        # Fit with simple data  
        cal_queries = ["Q1", "Q2", "Q3", "Q4", "Q5"]
        cal_responses = ["A1", "A2", "A3", "A4", "A5"]
        cal_truth = ["A1", "A2", "A3", "A4", "A5"]
        cal_confidence = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
        
        controller.fit(
            calibration_queries=cal_queries,
            calibration_responses=cal_responses,
            calibration_ground_truth=cal_truth,
            alpha=0.1,
            confidence_scores=cal_confidence
        )
        
        # Test scoring
        test_queries = ["TQ1", "TQ2"]
        test_responses = ["TR1", "TR2"]
        test_truth = ["TR1", "TR2"]
        test_confidence = np.array([0.8, 0.6])
        
        scores = controller.score(
            test_queries=test_queries,
            test_responses=test_responses,
            test_ground_truth=test_truth,
            confidence_scores=test_confidence
        )
        
        assert isinstance(scores, dict)
        assert 'empirical_hallucination_risk' in scores
        assert 'abstention_rate' in scores
        assert 'target_risk' in scores
        assert 'calibrated_lambda' in scores
        assert 'risk_controlled' in scores
        
    def test_get_risk_curve(self):
        """Test risk curve extraction."""
        loss_fn = WeightedHallucinationLoss()
        policy = AbstractionPolicy()
        controller = HallucinationController(loss_fn, policy, random_state=42)
        
        # Fit with simple data
        cal_queries = ["Q1", "Q2", "Q3"]
        cal_responses = ["A1", "A2", "A3"]
        cal_truth = ["A1", "A2", "A3"]
        cal_confidence = np.array([0.9, 0.8, 0.7])
        
        controller.fit(
            calibration_queries=cal_queries,
            calibration_responses=cal_responses,
            calibration_ground_truth=cal_truth,
            alpha=0.1,
            confidence_scores=cal_confidence
        )
        
        lambda_grid, risks = controller.get_risk_curve()
        
        assert len(lambda_grid) == len(risks)
        assert len(lambda_grid) > 0
        
    def test_set_alpha(self):
        """Test updating alpha value."""
        loss_fn = WeightedHallucinationLoss()
        policy = AbstractionPolicy() 
        controller = HallucinationController(loss_fn, policy, random_state=42)
        
        # Fit with simple data
        cal_queries = ["Q1", "Q2", "Q3"]
        cal_responses = ["A1", "A2", "A3"]
        cal_truth = ["A1", "A2", "A3"]
        cal_confidence = np.array([0.9, 0.8, 0.7])
        
        controller.fit(
            calibration_queries=cal_queries,
            calibration_responses=cal_responses,
            calibration_ground_truth=cal_truth,
            alpha=0.1,
            confidence_scores=cal_confidence
        )
        
        original_lambda = controller.lambda_star_
        
        # Update alpha
        controller.set_alpha(0.05)
        
        assert controller.alpha_ == 0.05
        # Lambda should typically change when alpha changes
        # (though not guaranteed in all cases)
        
    def test_with_different_lambda_grid(self):
        """Test controller with custom lambda grid."""
        loss_fn = WeightedHallucinationLoss()
        policy = AbstractionPolicy()
        custom_grid = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        
        controller = HallucinationController(
            loss_fn, policy, 
            lambda_grid=custom_grid,
            random_state=42
        )
        
        # Fit with simple data
        cal_queries = ["Q1", "Q2", "Q3"]
        cal_responses = ["A1", "A2", "A3"]
        cal_truth = ["A1", "A2", "A3"]
        cal_confidence = np.array([0.9, 0.8, 0.7])
        
        controller.fit(
            calibration_queries=cal_queries,
            calibration_responses=cal_responses,
            calibration_ground_truth=cal_truth,
            alpha=0.1,
            confidence_scores=cal_confidence
        )
        
        # Should use the custom grid
        assert np.array_equal(controller.lambda_grid, custom_grid)
        assert controller.lambda_star_ in custom_grid