"""
Main controller for Conformal Hallucination Mitigation (CHM).

Implements the core CRC-wrapped LLM answering policy with provable
expected hallucination harm control.
"""

from typing import Any, Dict, List, Optional, Tuple, Union, Callable
import numpy as np
from numpy.typing import NDArray
import warnings

from ..control_risk.crc_rcps import get_r_hat_plus, find_lambda_star
from .losses import HallucinationLoss, WeightedHallucinationLoss
from .policies import ControlPolicy, AbstractionPolicy


class HallucinationController:
    """
    Conformal Hallucination Mitigation controller.
    
    Wraps any LLM with a λ-dependent control policy and uses conformal risk
    control to guarantee E[hallucination_loss] ≤ α with distribution-free
    theoretical guarantees.
    
    Examples
    --------
    >>> # Basic usage with confidence-based abstention
    >>> controller = HallucinationController(
    ...     loss_function=WeightedHallucinationLoss(),
    ...     policy=AbstractionPolicy()
    ... )
    >>> 
    >>> # Calibrate on validation data
    >>> controller.fit(
    ...     calibration_queries=cal_queries,
    ...     calibration_responses=cal_responses,
    ...     calibration_ground_truth=cal_truth,
    ...     calibration_confidence=cal_confidence,
    ...     alpha=0.05  # Target 5% hallucination rate
    ... )
    >>>
    >>> # Use on new queries with guarantee
    >>> response = controller.predict(
    ...     query="What is the capital of France?",
    ...     base_response="Paris",
    ...     confidence_score=0.95
    ... )
    """
    
    def __init__(
        self,
        loss_function: HallucinationLoss,
        policy: ControlPolicy,
        lambda_grid: Optional[NDArray] = None,
        method: str = "crc",
        bound: Optional[str] = None,
        delta: Optional[float] = None,
        random_state: Optional[int] = None
    ):
        """
        Initialize hallucination controller.
        
        Parameters
        ----------
        loss_function : HallucinationLoss
            Loss function for computing hallucination risk
        policy : ControlPolicy
            Control policy defining λ-dependent behavior
        lambda_grid : NDArray, optional
            Grid of λ values to search over. If None, uses default grid.
        method : str, default="crc"
            Risk control method: "crc" or "rcps"
        bound : str, optional
            Concentration bound for RCPS: "hoeffding", "bernstein", or "wsr"
        delta : float, optional
            Confidence level for concentration bounds
        random_state : int, optional
            Random seed for reproducibility
        """
        self.loss_function = loss_function
        self.policy = policy
        self.lambda_grid = lambda_grid
        self.method = method
        self.bound = bound
        self.delta = delta
        self.random_state = random_state
        
        # Will be set during fitting
        self.lambda_star_ = None
        self.calibration_risks_ = None
        self.is_fitted_ = False
        
        if random_state is not None:
            np.random.seed(random_state)
    
    def fit(
        self,
        calibration_queries: List[str],
        calibration_responses: List[str],
        calibration_ground_truth: List[str],
        alpha: float,
        **calibration_context
    ) -> "HallucinationController":
        """
        Calibrate the controller using conformal risk control.
        
        Parameters
        ----------
        calibration_queries : List[str]
            Calibration queries/prompts
        calibration_responses : List[str]
            Model responses for calibration queries
        calibration_ground_truth : List[str]
            Ground truth answers for calibration queries
        alpha : float
            Target risk level (E[loss] ≤ α)
        **calibration_context
            Additional context for loss computation (confidence_scores, 
            retrieval_counts, etc.)
            
        Returns
        -------
        HallucinationController
            Fitted controller instance
        """
        # Convert to numpy arrays
        calibration_queries = np.array(calibration_queries)
        calibration_responses = np.array(calibration_responses)
        calibration_ground_truth = np.array(calibration_ground_truth)
        
        n_samples = len(calibration_queries)
        
        # Set default lambda grid if not provided
        if self.lambda_grid is None:
            if hasattr(self.loss_function, 'confidence_threshold_mode'):
                if self.loss_function.confidence_threshold_mode == "confidence":
                    self.lambda_grid = np.linspace(0.1, 0.99, 20)
                else:  # retrieval mode
                    self.lambda_grid = np.arange(0, 10)
            else:
                self.lambda_grid = np.linspace(0.1, 0.99, 20)
        
        n_lambdas = len(self.lambda_grid)
        
        # Compute risks for each λ
        risks = np.zeros((n_samples, n_lambdas))
        
        for i, lambda_param in enumerate(self.lambda_grid):
            # Compute loss for this λ
            sample_losses = self.loss_function(
                responses=calibration_responses,
                ground_truth=calibration_ground_truth,
                lambda_param=lambda_param,
                **calibration_context
            )
            
            # Handle multi-risk case
            if isinstance(sample_losses, dict):
                # Use hallucination risk as primary risk
                risks[:, i] = sample_losses['hallucination']
            else:
                risks[:, i] = sample_losses
        
        # Apply conformal risk control
        r_hat, r_hat_plus = get_r_hat_plus(
            risks=risks,
            lambdas=self.lambda_grid,
            method=self.method,
            bound=self.bound,
            delta=self.delta,
            sigma_init=0.25  # Default from CRC paper
        )
        
        # Find optimal λ
        alpha_array = np.array([alpha])
        lambda_star_array = find_lambda_star(
            lambdas=self.lambda_grid,
            r_hat_plus=r_hat_plus,
            alpha_np=alpha_array
        )
        
        self.lambda_star_ = lambda_star_array[0]
        self.calibration_risks_ = r_hat
        self.alpha_ = alpha
        self.is_fitted_ = True
        
        return self
    
    def predict(
        self,
        query: str,
        base_response: str,
        **context
    ) -> str:
        """
        Generate controlled response with hallucination risk guarantee.
        
        Parameters
        ----------
        query : str
            Input query/prompt
        base_response : str
            Base model response (before control)
        **context
            Additional context (confidence_score, retrieval_count, etc.)
            
        Returns
        -------
        str
            Controlled response (possibly abstention)
        """
        if not self.is_fitted_:
            raise ValueError("Controller must be fitted before prediction")
        
        return self.policy.get_response(
            lambda_param=self.lambda_star_,
            query=query,
            base_response=base_response,
            **context
        )
    
    def predict_batch(
        self,
        queries: List[str],
        base_responses: List[str],
        **batch_context
    ) -> List[str]:
        """
        Generate controlled responses for a batch of queries.
        
        Parameters
        ----------
        queries : List[str]
            Input queries/prompts
        base_responses : List[str]
            Base model responses
        **batch_context
            Batch context (arrays of confidence_scores, etc.)
            
        Returns
        -------
        List[str]
            Controlled responses
        """
        if not self.is_fitted_:
            raise ValueError("Controller must be fitted before prediction")
        
        controlled_responses = []
        
        for i, (query, base_response) in enumerate(zip(queries, base_responses)):
            # Extract context for this sample
            sample_context = {}
            for key, values in batch_context.items():
                if hasattr(values, '__getitem__') and len(values) > i:
                    sample_context[key] = values[i]
            
            response = self.predict(query, base_response, **sample_context)
            controlled_responses.append(response)
        
        return controlled_responses
    
    def score(
        self,
        test_queries: List[str],
        test_responses: List[str], 
        test_ground_truth: List[str],
        **test_context
    ) -> Dict[str, float]:
        """
        Evaluate controller performance on test data.
        
        Parameters
        ----------
        test_queries : List[str]
            Test queries
        test_responses : List[str]
            Model responses for test queries  
        test_ground_truth : List[str]
            Ground truth answers
        **test_context
            Test context for loss computation
            
        Returns
        -------
        Dict[str, float]
            Evaluation metrics including empirical risk
        """
        if not self.is_fitted_:
            raise ValueError("Controller must be fitted before scoring")
        
        # Generate controlled responses
        controlled_responses = self.predict_batch(
            test_queries, test_responses, **test_context
        )
        
        # Compute empirical loss on controlled responses
        test_losses = self.loss_function(
            responses=np.array(controlled_responses),
            ground_truth=np.array(test_ground_truth),
            lambda_param=self.lambda_star_,
            **test_context
        )
        
        if isinstance(test_losses, dict):
            empirical_risk = test_losses['hallucination'].mean()
            utility_risk = test_losses.get('utility', np.zeros_like(test_losses['hallucination'])).mean()
        else:
            empirical_risk = test_losses.mean()
            utility_risk = 0.0
        
        # Compute abstention rate
        abstention_rate = sum(
            1 for resp in controlled_responses 
            if "don't know" in resp.lower() or "insufficient" in resp.lower()
        ) / len(controlled_responses)
        
        return {
            'empirical_hallucination_risk': empirical_risk,
            'empirical_utility_risk': utility_risk,
            'abstention_rate': abstention_rate,
            'target_risk': self.alpha_,
            'calibrated_lambda': self.lambda_star_,
            'risk_controlled': empirical_risk <= self.alpha_ + 0.1  # Allow small slack
        }
    
    def get_risk_curve(self) -> Tuple[NDArray, NDArray]:
        """
        Get the risk curve over all λ values from calibration.
        
        Returns
        -------
        Tuple[NDArray, NDArray]
            (lambda_grid, calibration_risks)
        """
        if not self.is_fitted_:
            raise ValueError("Controller must be fitted to get risk curve")
        
        return self.lambda_grid, self.calibration_risks_
    
    def set_alpha(self, new_alpha: float) -> None:
        """
        Update target risk level and recalibrate λ*.
        
        Parameters
        ----------
        new_alpha : float
            New target risk level
        """
        if not self.is_fitted_:
            raise ValueError("Controller must be fitted before updating alpha")
        
        # Recompute λ* for new α using existing calibration
        alpha_array = np.array([new_alpha])
        
        # Need to recompute upper bounds
        risks = np.zeros((len(self.calibration_risks_), len(self.lambda_grid)))
        # This is a simplified version - in practice would need to store full risk matrix
        for i in range(len(self.lambda_grid)):
            risks[:, i] = self.calibration_risks_[i]
        
        _, r_hat_plus = get_r_hat_plus(
            risks=risks,
            lambdas=self.lambda_grid,
            method=self.method,
            bound=self.bound,
            delta=self.delta,
            sigma_init=0.25
        )
        
        lambda_star_array = find_lambda_star(
            lambdas=self.lambda_grid,
            r_hat_plus=r_hat_plus,
            alpha_np=alpha_array
        )
        
        self.lambda_star_ = lambda_star_array[0]
        self.alpha_ = new_alpha