"""
Tests for hallucination loss functions.
"""

import numpy as np
import pytest
from mapie.hallucination.losses import (
    WeightedHallucinationLoss,
    MultiRiskLoss,
    ConsistencyBasedLoss
)


class TestWeightedHallucinationLoss:
    """Test weighted hallucination loss function."""
    
    def test_basic_functionality(self):
        """Test basic loss computation."""
        loss_fn = WeightedHallucinationLoss()
        
        responses = np.array(["Paris", "London", "Berlin"])
        ground_truth = np.array(["Paris", "Paris", "Berlin"])
        confidence_scores = np.array([0.9, 0.3, 0.8])
        lambda_param = 0.5
        
        losses = loss_fn(
            responses=responses,
            ground_truth=ground_truth,
            lambda_param=lambda_param,
            confidence_scores=confidence_scores
        )
        
        # Should have one hallucination (London != Paris) and one abstention (0.3 < 0.5)
        assert len(losses) == 3
        assert all(loss >= 0 for loss in losses)
        
    def test_monotonicity_confidence(self):
        """Test that loss is monotone in confidence threshold."""
        loss_fn = WeightedHallucinationLoss()
        
        responses = np.array(["Wrong", "Also Wrong"])
        ground_truth = np.array(["Right", "Correct"])
        confidence_scores = np.array([0.6, 0.4])
        
        # Higher lambda should lead to more abstentions, potentially lower loss
        loss_low = loss_fn(responses, ground_truth, 0.3, confidence_scores=confidence_scores)
        loss_high = loss_fn(responses, ground_truth, 0.7, confidence_scores=confidence_scores)
        
        # At lambda=0.7, both should abstain (penalty only)
        # At lambda=0.3, both should answer (hallucination penalty)
        assert np.mean(loss_high) <= np.mean(loss_low) + 0.1  # Allow small tolerance
        
    def test_custom_weights(self):
        """Test custom severity weights."""
        weights = np.array([1.0, 5.0, 0.1])  # Middle sample has high severity
        loss_fn = WeightedHallucinationLoss(weights=weights)
        
        responses = np.array(["Wrong", "Wrong", "Wrong"])
        ground_truth = np.array(["Right", "Right", "Right"])
        confidence_scores = np.array([0.9, 0.9, 0.9])  # All confident
        lambda_param = 0.5
        
        losses = loss_fn(
            responses, ground_truth, lambda_param, 
            confidence_scores=confidence_scores
        )
        
        # Middle sample should have highest loss due to weight
        assert losses[1] > losses[0]
        assert losses[1] > losses[2]
        
    def test_retrieval_mode(self):
        """Test retrieval-based threshold mode."""
        loss_fn = WeightedHallucinationLoss(
            confidence_threshold_mode="retrieval"
        )
        
        responses = np.array(["Answer1", "Answer2"])
        ground_truth = np.array(["Answer1", "Answer2"])
        retrieval_counts = np.array([3, 1])
        lambda_param = 2  # Require at least 2 supporting documents
        
        losses = loss_fn(
            responses, ground_truth, lambda_param,
            retrieval_counts=retrieval_counts
        )
        
        # First should answer (3 >= 2), second should abstain (1 < 2)
        assert losses[0] < losses[1]


class TestMultiRiskLoss:
    """Test multi-risk loss function."""
    
    def test_multi_risk_output(self):
        """Test that multi-risk returns dictionary."""
        base_loss = WeightedHallucinationLoss()
        multi_loss = MultiRiskLoss(base_loss)
        
        responses = np.array(["Answer"])
        ground_truth = np.array(["Answer"])
        confidence_scores = np.array([0.8])
        
        result = multi_loss(
            responses, ground_truth, 0.5,
            confidence_scores=confidence_scores
        )
        
        assert isinstance(result, dict)
        assert 'hallucination' in result
        assert 'utility' in result
        assert len(result['hallucination']) == 1
        assert len(result['utility']) == 1


class TestConsistencyBasedLoss:
    """Test consistency-based loss function."""
    
    def test_consistency_loss(self):
        """Test consistency-based loss computation."""
        loss_fn = ConsistencyBasedLoss()
        
        responses = np.array(["Answer1", "Answer2"])
        ground_truth = np.array(["Answer1", "Answer2"])
        consistency_counts = np.array([3, 1])
        lambda_param = 2  # Require at least 2 consistent samples
        
        losses = loss_fn(
            responses, ground_truth, lambda_param,
            consistency_counts=consistency_counts
        )
        
        # First should answer (3 >= 2), second should abstain (1 < 2)
        assert len(losses) == 2
        assert all(loss >= 0 for loss in losses)
        
    def test_monotonicity_consistency(self):
        """Test monotonicity in consistency threshold."""
        loss_fn = ConsistencyBasedLoss()
        
        responses = np.array(["Wrong"])
        ground_truth = np.array(["Right"])
        consistency_counts = np.array([2])
        
        loss_low = loss_fn(responses, ground_truth, 1, consistency_counts=consistency_counts)
        loss_high = loss_fn(responses, ground_truth, 3, consistency_counts=consistency_counts)
        
        # Higher threshold should cause abstention instead of hallucination
        assert loss_high[0] <= loss_low[0]