"""
Benchmark framework for evaluating hallucination mitigation.

Provides datasets, evaluation metrics, and standardized protocols for
comparing different hallucination control approaches.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from numpy.typing import NDArray
import json
import random
from abc import ABC, abstractmethod


class HallucinationDataset(ABC):
    """
    Abstract base class for hallucination evaluation datasets.
    """
    
    @abstractmethod
    def get_samples(self, n_samples: Optional[int] = None) -> Dict[str, List]:
        """
        Get dataset samples.
        
        Parameters
        ----------
        n_samples : int, optional
            Number of samples to return. If None, returns all.
            
        Returns
        -------
        Dict[str, List]
            Dictionary with 'queries', 'ground_truth', and metadata
        """
        pass


class SyntheticQADataset(HallucinationDataset):
    """
    Synthetic question-answering dataset for testing hallucination control.
    
    Generates simple factual questions with known answers and controllable
    difficulty levels to test different aspects of hallucination mitigation.
    """
    
    def __init__(
        self,
        n_samples: int = 1000,
        difficulty_levels: List[str] = ["easy", "medium", "hard"],
        categories: List[str] = ["geography", "science", "history"],
        random_state: Optional[int] = None
    ):
        """
        Initialize synthetic dataset.
        
        Parameters
        ----------
        n_samples : int, default=1000
            Total number of samples to generate
        difficulty_levels : List[str]
            Difficulty levels to include
        categories : List[str]
            Question categories to include
        random_state : int, optional
            Random seed for reproducibility
        """
        self.n_samples = n_samples
        self.difficulty_levels = difficulty_levels
        self.categories = categories
        self.random_state = random_state
        
        if random_state is not None:
            random.seed(random_state)
            np.random.seed(random_state)
            
        self._generate_dataset()
        
    def _generate_dataset(self):
        """Generate synthetic QA samples."""
        # Pre-defined question templates and answers
        question_templates = {
            "geography": {
                "easy": [
                    ("What is the capital of France?", "Paris"),
                    ("What is the largest ocean?", "Pacific Ocean"),
                    ("Which continent is Egypt in?", "Africa"),
                ],
                "medium": [
                    ("What is the capital of Kazakhstan?", "Nur-Sultan"),
                    ("Which river flows through Baghdad?", "Tigris River"),
                    ("What is the smallest country in South America?", "Suriname"),
                ],
                "hard": [
                    ("What is the capital of Vanuatu?", "Port Vila"),
                    ("Which mountain range separates Europe from Asia?", "Ural Mountains"),
                    ("What is the second largest island in the Mediterranean?", "Sardinia"),
                ]
            },
            "science": {
                "easy": [
                    ("What gas do plants absorb during photosynthesis?", "Carbon dioxide"),
                    ("How many legs does a spider have?", "Eight"),
                    ("What is the chemical symbol for gold?", "Au"),
                ],
                "medium": [
                    ("What is the powerhouse of the cell?", "Mitochondria"),
                    ("What is the speed of light in vacuum?", "299,792,458 meters per second"),
                    ("Which element has atomic number 6?", "Carbon"),
                ],
                "hard": [
                    ("What is the Chandrasekhar limit?", "1.4 solar masses"),
                    ("Which enzyme unwinds DNA during replication?", "DNA helicase"),
                    ("What is the first law of thermodynamics?", "Energy cannot be created or destroyed"),
                ]
            },
            "history": {
                "easy": [
                    ("Who was the first president of the United States?", "George Washington"),
                    ("In which year did World War II end?", "1945"),
                    ("Which empire was ruled by Julius Caesar?", "Roman Empire"),
                ],
                "medium": [
                    ("Who wrote the Communist Manifesto?", "Karl Marx and Friedrich Engels"),
                    ("Which treaty ended World War I?", "Treaty of Versailles"),
                    ("Who was the first person to circumnavigate the globe?", "Ferdinand Magellan"),
                ],
                "hard": [
                    ("What was the Defenestration of Prague?", "1618 event that sparked the Thirty Years' War"),
                    ("Who was the last Byzantine Emperor?", "Constantine XI Palaiologos"),
                    ("What year was the Battle of Hastings?", "1066"),
                ]
            }
        }
        
        # Generate samples
        queries = []
        ground_truth = []
        difficulties = []
        categories_list = []
        
        samples_per_combo = self.n_samples // (len(self.difficulty_levels) * len(self.categories))
        
        for category in self.categories:
            for difficulty in self.difficulty_levels:
                templates = question_templates.get(category, {}).get(difficulty, [])
                
                for _ in range(samples_per_combo):
                    if templates:
                        question, answer = random.choice(templates)
                        queries.append(question)
                        ground_truth.append(answer)
                        difficulties.append(difficulty)
                        categories_list.append(category)
        
        # Pad to exact n_samples if needed
        while len(queries) < self.n_samples:
            category = random.choice(self.categories)
            difficulty = random.choice(self.difficulty_levels)
            templates = question_templates.get(category, {}).get(difficulty, [])
            if templates:
                question, answer = random.choice(templates)
                queries.append(question)
                ground_truth.append(answer)
                difficulties.append(difficulty)
                categories_list.append(category)
        
        self.queries = queries[:self.n_samples]
        self.ground_truth = ground_truth[:self.n_samples]
        self.difficulties = difficulties[:self.n_samples]
        self.categories_list = categories_list[:self.n_samples]
    
    def get_samples(self, n_samples: Optional[int] = None) -> Dict[str, List]:
        """Get dataset samples."""
        if n_samples is None:
            n_samples = self.n_samples
        else:
            n_samples = min(n_samples, self.n_samples)
            
        indices = list(range(n_samples))
        
        return {
            'queries': [self.queries[i] for i in indices],
            'ground_truth': [self.ground_truth[i] for i in indices],
            'difficulties': [self.difficulties[i] for i in indices],
            'categories': [self.categories_list[i] for i in indices]
        }


class MockLLMSimulator:
    """
    Mock LLM simulator for testing hallucination control.
    
    Simulates an LLM with controllable hallucination rates, confidence
    calibration, and retrieval capabilities.
    """
    
    def __init__(
        self,
        base_accuracy: float = 0.8,
        confidence_noise: float = 0.1,
        hallucination_rate: float = 0.15,
        random_state: Optional[int] = None
    ):
        """
        Initialize mock LLM.
        
        Parameters
        ----------
        base_accuracy : float, default=0.8
            Base accuracy when answering questions
        confidence_noise : float, default=0.1
            Noise in confidence calibration
        hallucination_rate : float, default=0.15
            Rate of confident hallucinations
        random_state : int, optional
            Random seed
        """
        self.base_accuracy = base_accuracy
        self.confidence_noise = confidence_noise
        self.hallucination_rate = hallucination_rate
        self.random_state = random_state
        
        if random_state is not None:
            np.random.seed(random_state)
    
    def generate_responses(
        self,
        queries: List[str],
        ground_truth: List[str],
        difficulties: Optional[List[str]] = None
    ) -> Dict[str, NDArray]:
        """
        Generate mock LLM responses with confidence scores.
        
        Parameters
        ----------
        queries : List[str]
            Input queries
        ground_truth : List[str]
            True answers
        difficulties : List[str], optional
            Question difficulties
            
        Returns
        -------
        Dict[str, NDArray]
            Dictionary with 'responses', 'confidence_scores', 'is_correct'
        """
        n_samples = len(queries)
        responses = []
        confidence_scores = []
        is_correct = []
        
        for i in range(n_samples):
            # Adjust accuracy based on difficulty
            if difficulties:
                difficulty = difficulties[i]
                if difficulty == "easy":
                    accuracy = self.base_accuracy + 0.1
                elif difficulty == "hard":
                    accuracy = self.base_accuracy - 0.2
                else:
                    accuracy = self.base_accuracy
            else:
                accuracy = self.base_accuracy
            
            # Determine if response is correct
            correct = np.random.random() < accuracy
            
            # Generate response
            if correct:
                response = ground_truth[i]
                base_confidence = 0.8
            else:
                # Generate plausible but wrong answer
                if "capital" in queries[i].lower():
                    wrong_capitals = ["London", "Berlin", "Madrid", "Rome", "Vienna"]
                    response = np.random.choice(wrong_capitals)
                elif "ocean" in queries[i].lower():
                    response = "Atlantic Ocean"
                else:
                    response = "I'm not sure about this."
                
                # Hallucination: sometimes wrong answers have high confidence
                if np.random.random() < self.hallucination_rate:
                    base_confidence = 0.7  # Confident hallucination
                else:
                    base_confidence = 0.3  # Uncertain wrong answer
            
            # Add noise to confidence
            confidence = np.clip(
                base_confidence + np.random.normal(0, self.confidence_noise),
                0.0, 1.0
            )
            
            responses.append(response)
            confidence_scores.append(confidence)
            is_correct.append(correct)
        
        return {
            'responses': np.array(responses),
            'confidence_scores': np.array(confidence_scores),
            'is_correct': np.array(is_correct)
        }


class HallucinationBenchmark:
    """
    Comprehensive benchmark for hallucination mitigation methods.
    
    Provides standardized evaluation protocols, datasets, and metrics
    for comparing different approaches to hallucination control.
    """
    
    def __init__(
        self,
        dataset: Optional[HallucinationDataset] = None,
        llm_simulator: Optional[MockLLMSimulator] = None,
        random_state: Optional[int] = None
    ):
        """
        Initialize benchmark.
        
        Parameters
        ----------
        dataset : HallucinationDataset, optional
            Dataset for evaluation. If None, uses default synthetic dataset.
        llm_simulator : MockLLMSimulator, optional
            LLM simulator. If None, uses default simulator.
        random_state : int, optional
            Random seed
        """
        self.dataset = dataset or SyntheticQADataset(random_state=random_state)
        self.llm_simulator = llm_simulator or MockLLMSimulator(random_state=random_state)
        self.random_state = random_state
    
    def evaluate_controller(
        self,
        controller: "HallucinationController",
        test_split: float = 0.5,
        alpha_values: List[float] = [0.05, 0.1, 0.15, 0.2]
    ) -> Dict[str, Any]:
        """
        Comprehensive evaluation of a hallucination controller.
        
        Parameters
        ----------
        controller : HallucinationController
            Controller to evaluate
        test_split : float, default=0.5
            Fraction of data to use for testing (rest for calibration)
        alpha_values : List[float]
            Risk levels to test
            
        Returns
        -------
        Dict[str, Any]
            Comprehensive evaluation results
        """
        # Get dataset samples
        all_samples = self.dataset.get_samples()
        n_total = len(all_samples['queries'])
        n_cal = int(n_total * (1 - test_split))
        
        # Split data
        cal_samples = {
            key: values[:n_cal] for key, values in all_samples.items()
        }
        test_samples = {
            key: values[n_cal:] for key, values in all_samples.items()
        }
        
        # Generate LLM responses for calibration
        cal_llm_output = self.llm_simulator.generate_responses(
            cal_samples['queries'],
            cal_samples['ground_truth'],
            cal_samples.get('difficulties')
        )
        
        # Generate LLM responses for test
        test_llm_output = self.llm_simulator.generate_responses(
            test_samples['queries'],
            test_samples['ground_truth'],
            test_samples.get('difficulties')
        )
        
        results = {}
        
        # Evaluate for each alpha value
        for alpha in alpha_values:
            # Fit controller
            controller.fit(
                calibration_queries=cal_samples['queries'],
                calibration_responses=cal_llm_output['responses'].tolist(),
                calibration_ground_truth=cal_samples['ground_truth'],
                alpha=alpha,
                confidence_scores=cal_llm_output['confidence_scores']
            )
            
            # Evaluate on test set
            test_scores = controller.score(
                test_queries=test_samples['queries'],
                test_responses=test_llm_output['responses'].tolist(),
                test_ground_truth=test_samples['ground_truth'],
                confidence_scores=test_llm_output['confidence_scores']
            )
            
            results[f'alpha_{alpha}'] = test_scores
        
        # Add baseline metrics (uncontrolled LLM)
        baseline_accuracy = test_llm_output['is_correct'].mean()
        baseline_hallucination_rate = 1 - baseline_accuracy
        
        results['baseline'] = {
            'accuracy': baseline_accuracy,
            'hallucination_rate': baseline_hallucination_rate,
            'abstention_rate': 0.0
        }
        
        return results
    
    def compare_controllers(
        self,
        controllers: Dict[str, "HallucinationController"],
        alpha: float = 0.1
    ) -> Dict[str, Dict[str, Any]]:
        """
        Compare multiple controllers head-to-head.
        
        Parameters
        ----------
        controllers : Dict[str, HallucinationController]
            Named controllers to compare
        alpha : float, default=0.1
            Risk level for comparison
            
        Returns
        -------
        Dict[str, Dict[str, Any]]
            Comparison results for each controller
        """
        results = {}
        
        for name, controller in controllers.items():
            controller_results = self.evaluate_controller(
                controller, alpha_values=[alpha]
            )
            results[name] = controller_results[f'alpha_{alpha}']
        
        return results
    
    def generate_report(
        self,
        evaluation_results: Dict[str, Any],
        save_path: Optional[str] = None
    ) -> str:
        """
        Generate a human-readable evaluation report.
        
        Parameters
        ----------
        evaluation_results : Dict[str, Any]
            Results from evaluate_controller or compare_controllers
        save_path : str, optional
            Path to save report
            
        Returns
        -------
        str
            Formatted report text
        """
        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("HALLUCINATION MITIGATION EVALUATION REPORT")
        report_lines.append("=" * 60)
        report_lines.append("")
        
        # Baseline performance
        if 'baseline' in evaluation_results:
            baseline = evaluation_results['baseline']
            report_lines.append("BASELINE (UNCONTROLLED) PERFORMANCE:")
            report_lines.append(f"  Accuracy: {baseline['accuracy']:.3f}")
            report_lines.append(f"  Hallucination Rate: {baseline['hallucination_rate']:.3f}")
            report_lines.append("")
        
        # Controlled performance for each alpha
        report_lines.append("CONTROLLED PERFORMANCE:")
        
        for key, results in evaluation_results.items():
            if key.startswith('alpha_'):
                alpha = key.replace('alpha_', '')
                report_lines.append(f"\n  Target Risk Level α = {alpha}:")
                report_lines.append(f"    Empirical Hallucination Risk: {results['empirical_hallucination_risk']:.3f}")
                report_lines.append(f"    Abstention Rate: {results['abstention_rate']:.3f}")
                report_lines.append(f"    Risk Controlled: {results['risk_controlled']}")
                report_lines.append(f"    Calibrated λ: {results['calibrated_lambda']:.3f}")
        
        report_text = "\n".join(report_lines)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
        
        return report_text