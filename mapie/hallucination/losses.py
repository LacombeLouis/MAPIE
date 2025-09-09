"""
Loss functions for hallucination risk control.

Implements various monotone loss functions that can be used with conformal risk control
to bound expected hallucination harm.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import numpy as np
from numpy.typing import NDArray


class HallucinationLoss(ABC):
    """
    Abstract base class for hallucination loss functions.
    
    Loss functions must be monotone in the control parameter λ to ensure
    conformal risk control guarantees hold.
    """
    
    @abstractmethod
    def __call__(
        self,
        responses: NDArray,
        ground_truth: NDArray,
        lambda_param: float,
        **kwargs
    ) -> NDArray:
        """
        Compute loss for each sample given control parameter λ.
        
        Parameters
        ----------
        responses : NDArray
            Model responses/outputs for each sample
        ground_truth : NDArray  
            True labels or reference answers
        lambda_param : float
            Control parameter (must result in monotone loss)
        **kwargs
            Additional parameters for loss computation
            
        Returns
        -------
        NDArray
            Loss value for each sample
        """
        pass


class WeightedHallucinationLoss(HallucinationLoss):
    """
    Weighted binary hallucination loss with abstention option.
    
    L_i(λ) = w_i * H_i(λ) + β * A_i(λ)
    
    where:
    - H_i(λ) = 1 if response i contains hallucination at threshold λ
    - A_i(λ) = 1 if response i results in abstention at threshold λ  
    - w_i = severity weight for sample i
    - β = small abstention penalty (β << min(w_i))
    """
    
    def __init__(
        self, 
        weights: Optional[NDArray] = None,
        abstention_penalty: float = 0.01,
        confidence_threshold_mode: str = "confidence"
    ):
        """
        Initialize weighted hallucination loss.
        
        Parameters
        ----------
        weights : NDArray, optional
            Severity weights for each sample. If None, uses uniform weights.
        abstention_penalty : float, default=0.01
            Small penalty for abstention to maintain monotonicity
        confidence_threshold_mode : str, default="confidence"
            How to interpret λ parameter:
            - "confidence": λ is minimum confidence threshold to answer
            - "retrieval": λ is minimum number of supporting documents
        """
        self.weights = weights
        self.abstention_penalty = abstention_penalty
        self.confidence_threshold_mode = confidence_threshold_mode
        
    def __call__(
        self,
        responses: NDArray,
        ground_truth: NDArray, 
        lambda_param: float,
        confidence_scores: Optional[NDArray] = None,
        retrieval_counts: Optional[NDArray] = None,
        verifier_scores: Optional[NDArray] = None
    ) -> NDArray:
        """
        Compute weighted hallucination loss.
        
        Parameters
        ----------
        responses : NDArray of shape (n_samples,)
            Model responses (strings or encoded)
        ground_truth : NDArray of shape (n_samples,)
            Ground truth answers
        lambda_param : float
            Control parameter threshold
        confidence_scores : NDArray, optional
            Model confidence scores for each response
        retrieval_counts : NDArray, optional
            Number of supporting documents retrieved for each query
        verifier_scores : NDArray, optional
            External verifier scores (probability response is correct)
            
        Returns
        -------
        NDArray of shape (n_samples,)
            Loss for each sample
        """
        n_samples = len(responses)
        
        # Initialize weights if not provided
        if self.weights is None:
            weights = np.ones(n_samples)
        else:
            weights = self.weights
            
        # Determine abstention based on threshold mode
        if self.confidence_threshold_mode == "confidence":
            if confidence_scores is None:
                raise ValueError("confidence_scores required for confidence mode")
            abstained = confidence_scores < lambda_param
        elif self.confidence_threshold_mode == "retrieval":
            if retrieval_counts is None:
                raise ValueError("retrieval_counts required for retrieval mode")
            abstained = retrieval_counts < lambda_param
        else:
            raise ValueError(f"Unknown threshold mode: {self.confidence_threshold_mode}")
        
        # For non-abstained responses, check for hallucinations
        hallucinated = np.zeros(n_samples, dtype=bool)
        for i in range(n_samples):
            if not abstained[i]:
                # Simple hallucination detection (can be replaced with sophisticated verifier)
                if verifier_scores is not None:
                    # Use external verifier
                    hallucinated[i] = verifier_scores[i] < 0.5
                else:
                    # Simple string comparison fallback
                    hallucinated[i] = str(responses[i]).lower() != str(ground_truth[i]).lower()
        
        # Compute loss: w_i * H_i(λ) + β * A_i(λ)
        loss = weights * hallucinated + self.abstention_penalty * abstained
        
        return loss


class MultiRiskLoss(HallucinationLoss):
    """
    Multi-objective loss combining hallucination risk and utility.
    
    Allows simultaneous control of:
    - Expected hallucination harm ≤ α_h
    - Expected abstention rate ≤ α_u (utility preservation)
    """
    
    def __init__(
        self,
        hallucination_loss: HallucinationLoss,
        utility_weight: float = 1.0
    ):
        """
        Initialize multi-risk loss.
        
        Parameters
        ----------
        hallucination_loss : HallucinationLoss
            Base hallucination loss function
        utility_weight : float, default=1.0
            Weight for utility preservation (abstention cost)
        """
        self.hallucination_loss = hallucination_loss
        self.utility_weight = utility_weight
        
    def __call__(
        self,
        responses: NDArray,
        ground_truth: NDArray,
        lambda_param: float,
        **kwargs
    ) -> Dict[str, NDArray]:
        """
        Compute both hallucination and utility losses.
        
        Returns
        -------
        Dict[str, NDArray]
            Dictionary with 'hallucination' and 'utility' losses
        """
        # Get base hallucination loss
        base_loss = self.hallucination_loss(
            responses, ground_truth, lambda_param, **kwargs
        )
        
        # Extract abstention indicators for utility loss
        confidence_scores = kwargs.get('confidence_scores')
        if confidence_scores is not None:
            abstained = confidence_scores < lambda_param
            utility_loss = self.utility_weight * abstained.astype(float)
        else:
            # Fallback: assume no abstention
            utility_loss = np.zeros(len(responses))
            
        return {
            'hallucination': base_loss,
            'utility': utility_loss
        }


class ConsistencyBasedLoss(HallucinationLoss):
    """
    Loss based on self-consistency across multiple model samples.
    
    Uses λ as the minimum number of consistent samples required before
    committing to an answer.
    """
    
    def __init__(self, consistency_penalty: float = 0.1):
        """
        Initialize consistency-based loss.
        
        Parameters
        ----------
        consistency_penalty : float, default=0.1
            Penalty for abstaining due to inconsistency
        """
        self.consistency_penalty = consistency_penalty
        
    def __call__(
        self,
        responses: NDArray,
        ground_truth: NDArray,
        lambda_param: float,
        consistency_counts: Optional[NDArray] = None,
        **kwargs
    ) -> NDArray:
        """
        Compute consistency-based loss.
        
        Parameters
        ----------
        responses : NDArray
            Model responses
        ground_truth : NDArray
            Ground truth answers  
        lambda_param : float
            Minimum consistency threshold (number of agreeing samples)
        consistency_counts : NDArray, optional
            Number of samples that agree with each response
            
        Returns
        -------
        NDArray
            Loss for each sample
        """
        n_samples = len(responses)
        
        if consistency_counts is None:
            # Default: assume no consistency information
            consistency_counts = np.ones(n_samples)
            
        # Abstain if consistency is below threshold
        abstained = consistency_counts < lambda_param
        
        # Check hallucinations for non-abstained responses
        hallucinated = np.zeros(n_samples, dtype=bool)
        for i in range(n_samples):
            if not abstained[i]:
                hallucinated[i] = str(responses[i]).lower() != str(ground_truth[i]).lower()
                
        # Loss: hallucination indicator + consistency penalty for abstention
        loss = hallucinated.astype(float) + self.consistency_penalty * abstained
        
        return loss