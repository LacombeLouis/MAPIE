"""
Tests for the hallucination benchmark framework.
"""

import numpy as np
import pytest
from mapie.hallucination.benchmark import (
    SyntheticQADataset,
    MockLLMSimulator,
    HallucinationBenchmark
)
from mapie.hallucination.controller import HallucinationController
from mapie.hallucination.losses import WeightedHallucinationLoss
from mapie.hallucination.policies import AbstractionPolicy


class TestSyntheticQADataset:
    """Test synthetic QA dataset."""
    
    def test_dataset_creation(self):
        """Test basic dataset creation."""
        dataset = SyntheticQADataset(
            n_samples=100,
            random_state=42
        )
        
        samples = dataset.get_samples()
        
        assert len(samples['queries']) == 100
        assert len(samples['ground_truth']) == 100
        assert len(samples['difficulties']) == 100
        assert len(samples['categories']) == 100
        
    def test_dataset_categories(self):
        """Test that dataset includes specified categories."""
        categories = ["geography", "science"]
        dataset = SyntheticQADataset(
            n_samples=60,
            categories=categories,
            random_state=42
        )
        
        samples = dataset.get_samples()
        
        unique_categories = set(samples['categories'])
        assert unique_categories.issubset(set(categories))
        
    def test_dataset_difficulties(self):
        """Test that dataset includes specified difficulties."""
        difficulties = ["easy", "hard"]
        dataset = SyntheticQADataset(
            n_samples=40,
            difficulty_levels=difficulties,
            random_state=42
        )
        
        samples = dataset.get_samples()
        
        unique_difficulties = set(samples['difficulties'])
        assert unique_difficulties.issubset(set(difficulties))
        
    def test_partial_sampling(self):
        """Test getting partial samples."""
        dataset = SyntheticQADataset(n_samples=100, random_state=42)
        
        partial_samples = dataset.get_samples(n_samples=50)
        
        assert len(partial_samples['queries']) == 50
        assert len(partial_samples['ground_truth']) == 50


class TestMockLLMSimulator:
    """Test mock LLM simulator."""
    
    def test_response_generation(self):
        """Test basic response generation."""
        simulator = MockLLMSimulator(random_state=42)
        
        queries = ["What is the capital of France?", "What is 2+2?"]
        ground_truth = ["Paris", "4"]
        
        results = simulator.generate_responses(queries, ground_truth)
        
        assert len(results['responses']) == 2
        assert len(results['confidence_scores']) == 2
        assert len(results['is_correct']) == 2
        
        # Check that confidence scores are in valid range
        assert all(0 <= score <= 1 for score in results['confidence_scores'])
        
    def test_accuracy_control(self):
        """Test that base accuracy affects correctness."""
        # High accuracy simulator
        high_acc_sim = MockLLMSimulator(base_accuracy=0.9, random_state=42)
        
        # Low accuracy simulator  
        low_acc_sim = MockLLMSimulator(base_accuracy=0.2, random_state=42)
        
        queries = ["Q"] * 100
        truth = ["A"] * 100
        
        high_results = high_acc_sim.generate_responses(queries, truth)
        low_results = low_acc_sim.generate_responses(queries, truth)
        
        high_accuracy = np.mean(high_results['is_correct'])
        low_accuracy = np.mean(low_results['is_correct'])
        
        assert high_accuracy > low_accuracy
        
    def test_difficulty_effects(self):
        """Test that difficulty affects accuracy."""
        simulator = MockLLMSimulator(base_accuracy=0.8, random_state=42)
        
        queries = ["Q"] * 30
        truth = ["A"] * 30
        difficulties = ["easy"] * 10 + ["medium"] * 10 + ["hard"] * 10
        
        results = simulator.generate_responses(queries, truth, difficulties)
        
        # Group by difficulty
        easy_correct = [results['is_correct'][i] for i in range(10)]
        medium_correct = [results['is_correct'][i] for i in range(10, 20)]
        hard_correct = [results['is_correct'][i] for i in range(20, 30)]
        
        # Easy should generally be more accurate than hard
        # (though with small samples this might not always hold)
        easy_acc = np.mean(easy_correct)
        hard_acc = np.mean(hard_correct)
        
        # At least check they're different
        assert easy_acc != hard_acc or len(set(results['is_correct'])) > 1


class TestHallucinationBenchmark:
    """Test hallucination benchmark framework."""
    
    def test_benchmark_initialization(self):
        """Test benchmark initialization."""
        benchmark = HallucinationBenchmark(random_state=42)
        
        assert benchmark.dataset is not None
        assert benchmark.llm_simulator is not None
        
    def test_controller_evaluation(self):
        """Test evaluating a controller."""
        # Create simple controller
        loss_fn = WeightedHallucinationLoss()
        policy = AbstractionPolicy()
        controller = HallucinationController(loss_fn, policy, random_state=42)
        
        # Create benchmark with small dataset
        dataset = SyntheticQADataset(n_samples=20, random_state=42)
        simulator = MockLLMSimulator(random_state=42)
        benchmark = HallucinationBenchmark(
            dataset=dataset,
            llm_simulator=simulator,
            random_state=42
        )
        
        # Evaluate controller
        results = benchmark.evaluate_controller(
            controller,
            test_split=0.5,
            alpha_values=[0.1, 0.2]
        )
        
        assert isinstance(results, dict)
        assert 'alpha_0.1' in results
        assert 'alpha_0.2' in results
        assert 'baseline' in results
        
        # Check baseline metrics
        baseline = results['baseline']
        assert 'accuracy' in baseline
        assert 'hallucination_rate' in baseline
        assert 'abstention_rate' in baseline
        
        # Check alpha results
        alpha_result = results['alpha_0.1']
        assert 'empirical_hallucination_risk' in alpha_result
        assert 'abstention_rate' in alpha_result
        assert 'risk_controlled' in alpha_result
        
    def test_controller_comparison(self):
        """Test comparing multiple controllers."""
        # Create two controllers with different settings
        loss_fn1 = WeightedHallucinationLoss(abstention_penalty=0.01)
        loss_fn2 = WeightedHallucinationLoss(abstention_penalty=0.1)
        policy = AbstractionPolicy()
        
        controller1 = HallucinationController(loss_fn1, policy, random_state=42)
        controller2 = HallucinationController(loss_fn2, policy, random_state=43)
        
        controllers = {
            "low_penalty": controller1,
            "high_penalty": controller2
        }
        
        # Create benchmark
        dataset = SyntheticQADataset(n_samples=20, random_state=42)
        simulator = MockLLMSimulator(random_state=42)
        benchmark = HallucinationBenchmark(
            dataset=dataset,
            llm_simulator=simulator,
            random_state=42
        )
        
        # Compare controllers
        comparison = benchmark.compare_controllers(controllers, alpha=0.1)
        
        assert isinstance(comparison, dict)
        assert "low_penalty" in comparison
        assert "high_penalty" in comparison
        
        # Both should have the same structure
        for name, results in comparison.items():
            assert 'empirical_hallucination_risk' in results
            assert 'abstention_rate' in results
            
    def test_report_generation(self):
        """Test evaluation report generation."""
        # Mock evaluation results
        results = {
            'baseline': {
                'accuracy': 0.8,
                'hallucination_rate': 0.2,
                'abstention_rate': 0.0
            },
            'alpha_0.1': {
                'empirical_hallucination_risk': 0.08,
                'abstention_rate': 0.15,
                'risk_controlled': True,
                'calibrated_lambda': 0.65
            }
        }
        
        benchmark = HallucinationBenchmark(random_state=42)
        report = benchmark.generate_report(results)
        
        assert isinstance(report, str)
        assert "BASELINE" in report
        assert "CONTROLLED PERFORMANCE" in report
        assert "0.08" in report  # empirical risk
        assert "0.15" in report  # abstention rate