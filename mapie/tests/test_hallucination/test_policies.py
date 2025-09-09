"""
Tests for hallucination control policies.
"""

import numpy as np
import pytest
from mapie.hallucination.policies import (
    AbstractionPolicy,
    RetrievalPolicy,
    ConsistencyPolicy,
    MultiParameterPolicy
)


class TestAbstractionPolicy:
    """Test confidence-based abstention policy."""
    
    def test_should_answer_with_confidence(self):
        """Test answering decision based on confidence."""
        policy = AbstractionPolicy()
        
        # Should answer when confidence >= threshold
        assert policy.should_answer(0.5, "query", confidence_score=0.8)
        assert not policy.should_answer(0.5, "query", confidence_score=0.3)
        
    def test_should_answer_without_confidence(self):
        """Test conservative behavior without confidence."""
        policy = AbstractionPolicy()
        
        # Should abstain when no confidence provided
        assert not policy.should_answer(0.5, "query")
        
    def test_get_response(self):
        """Test response generation."""
        policy = AbstractionPolicy(abstention_message="Uncertain")
        
        # Should return base response when confident
        response = policy.get_response(
            0.5, "query", "base response", confidence_score=0.8
        )
        assert response == "base response"
        
        # Should return abstention message when not confident
        response = policy.get_response(
            0.5, "query", "base response", confidence_score=0.3
        )
        assert response == "Uncertain"


class TestRetrievalPolicy:
    """Test retrieval-based policy."""
    
    def test_should_answer_with_count(self):
        """Test answering decision based on retrieval count."""
        policy = RetrievalPolicy()
        
        # Should answer when sufficient evidence
        assert policy.should_answer(2, "query", retrieval_count=3)
        assert not policy.should_answer(2, "query", retrieval_count=1)
        
    def test_should_answer_with_docs(self):
        """Test answering decision based on retrieved documents."""
        policy = RetrievalPolicy()
        
        docs = ["doc1", "doc2", "doc3"]
        assert policy.should_answer(2, "query", retrieved_docs=docs)
        assert not policy.should_answer(5, "query", retrieved_docs=docs)
        
    def test_get_response_with_evidence(self):
        """Test response enhancement with evidence."""
        policy = RetrievalPolicy()
        
        response = policy.get_response(
            2, "query", "base response",
            retrieval_count=3,
            retrieved_docs=["doc1", "doc2", "doc3"]
        )
        
        assert "base response" in response
        assert "3 supporting sources" in response


class TestConsistencyPolicy:
    """Test self-consistency policy."""
    
    def test_should_answer_with_count(self):
        """Test answering decision based on consistency count."""
        policy = ConsistencyPolicy()
        
        assert policy.should_answer(3, "query", consistency_count=4)
        assert not policy.should_answer(3, "query", consistency_count=2)
        
    def test_should_answer_with_samples(self):
        """Test answering decision based on sample responses."""
        policy = ConsistencyPolicy()
        
        # Majority agrees on "Paris"
        samples = ["Paris", "Paris", "Paris", "London"]
        assert policy.should_answer(3, "query", sample_responses=samples)
        assert not policy.should_answer(4, "query", sample_responses=samples)
        
    def test_compute_agreement(self):
        """Test agreement computation."""
        policy = ConsistencyPolicy()
        
        samples = ["A", "A", "B", "A"]
        agreement = policy._compute_agreement(samples)
        assert agreement == 3  # Three "A"s
        
    def test_get_consensus(self):
        """Test consensus response generation."""
        policy = ConsistencyPolicy()
        
        samples = ["Paris", "Paris", "London"]
        consensus = policy._get_consensus(samples)
        assert consensus == "Paris"


class TestMultiParameterPolicy:
    """Test multi-parameter policy."""
    
    def test_initialization(self):
        """Test policy initialization with sub-policies."""
        conf_policy = AbstractionPolicy()
        retr_policy = RetrievalPolicy()
        
        multi_policy = MultiParameterPolicy(
            confidence_policy=conf_policy,
            retrieval_policy=retr_policy
        )
        
        assert multi_policy.confidence_policy is conf_policy
        assert multi_policy.retrieval_policy is retr_policy
        
    def test_should_answer_all_mode(self):
        """Test 'all' combination mode."""
        policy = MultiParameterPolicy(combination_mode="all")
        
        # All policies must agree
        result = policy.should_answer(
            (0.5, 2, 2),  # (confidence, retrieval, consistency)
            "query",
            confidence_score=0.8,  # Pass confidence
            retrieval_count=3,     # Pass retrieval  
            consistency_count=1    # Fail consistency
        )
        assert not result  # Should fail because consistency fails
        
    def test_should_answer_any_mode(self):
        """Test 'any' combination mode."""
        policy = MultiParameterPolicy(combination_mode="any")
        
        # Any policy can approve
        result = policy.should_answer(
            (0.5, 2, 2),
            "query",
            confidence_score=0.8,  # Pass confidence
            retrieval_count=1,     # Fail retrieval
            consistency_count=1    # Fail consistency
        )
        assert result  # Should pass because confidence passes
        
    def test_should_answer_dict_params(self):
        """Test dictionary parameter format."""
        policy = MultiParameterPolicy()
        
        lambda_dict = {
            "confidence": 0.5,
            "retrieval": 2,
            "consistency": 2
        }
        
        result = policy.should_answer(
            lambda_dict,
            "query",
            confidence_score=0.8,
            retrieval_count=3,
            consistency_count=3
        )
        assert result  # All should pass
        
    def test_get_response_with_enhancements(self):
        """Test response enhancement from multiple policies."""
        policy = MultiParameterPolicy()
        
        response = policy.get_response(
            0.5,  # Use same threshold for all
            "query",
            "base response",
            confidence_score=0.8,
            retrieval_count=3,
            retrieved_docs=["doc1", "doc2", "doc3"],
            consistency_count=2
        )
        
        assert "base response" in response
        # Should have enhancements from evidence and consistency
        assert "3 sources" in response or "2 samples" in response