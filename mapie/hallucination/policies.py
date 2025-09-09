"""
Control policies for hallucination mitigation.

Implements various λ-parameterized policies that control LLM behavior
to reduce hallucination risk while maintaining utility.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from numpy.typing import NDArray


class ControlPolicy(ABC):
    """
    Abstract base class for hallucination control policies.
    
    Policies define how the control parameter λ affects LLM behavior
    (e.g., abstention thresholds, retrieval requirements, etc.)
    """
    
    @abstractmethod
    def should_answer(
        self,
        lambda_param: float,
        query: str,
        **context
    ) -> bool:
        """
        Determine whether to answer a query given control parameter λ.
        
        Parameters
        ----------
        lambda_param : float
            Control parameter value
        query : str
            Input query/prompt
        **context
            Additional context (confidence, retrieval results, etc.)
            
        Returns
        -------
        bool
            True if should answer, False if should abstain
        """
        pass
    
    @abstractmethod
    def get_response(
        self,
        lambda_param: float,
        query: str,
        base_response: str,
        **context
    ) -> str:
        """
        Generate final response given control parameter λ.
        
        Parameters
        ----------
        lambda_param : float
            Control parameter value
        query : str
            Input query/prompt
        base_response : str
            Base model response
        **context
            Additional context
            
        Returns
        -------
        str
            Final response (possibly modified or abstention message)
        """
        pass


class AbstractionPolicy(ControlPolicy):
    """
    Simple confidence-based abstention policy.
    
    Abstains from answering when model confidence is below λ threshold.
    """
    
    def __init__(self, abstention_message: str = "I don't know."):
        """
        Initialize abstention policy.
        
        Parameters
        ----------
        abstention_message : str, default="I don't know."
            Message to return when abstaining
        """
        self.abstention_message = abstention_message
        
    def should_answer(
        self,
        lambda_param: float,
        query: str,
        confidence_score: Optional[float] = None,
        **context
    ) -> bool:
        """
        Answer only if confidence exceeds λ threshold.
        
        Parameters
        ----------
        lambda_param : float
            Minimum confidence threshold
        query : str
            Input query
        confidence_score : float, optional
            Model confidence score
        **context
            Additional context
            
        Returns
        -------
        bool
            True if confidence >= λ, False otherwise
        """
        if confidence_score is None:
            # Conservative: abstain if no confidence available
            return False
        return confidence_score >= lambda_param
    
    def get_response(
        self,
        lambda_param: float,
        query: str,
        base_response: str,
        **context
    ) -> str:
        """
        Return base response if should answer, otherwise abstain.
        """
        if self.should_answer(lambda_param, query, **context):
            return base_response
        else:
            return self.abstention_message


class RetrievalPolicy(ControlPolicy):
    """
    Retrieval-based policy that requires minimum evidence.
    
    Only answers when sufficient supporting documents are retrieved.
    """
    
    def __init__(
        self,
        retrieval_function: Optional[callable] = None,
        abstention_message: str = "Insufficient evidence to answer."
    ):
        """
        Initialize retrieval policy.
        
        Parameters
        ----------
        retrieval_function : callable, optional
            Function to retrieve supporting documents
        abstention_message : str
            Message when insufficient evidence
        """
        self.retrieval_function = retrieval_function
        self.abstention_message = abstention_message
        
    def should_answer(
        self,
        lambda_param: float,
        query: str,
        retrieval_count: Optional[int] = None,
        retrieved_docs: Optional[List[str]] = None,
        **context
    ) -> bool:
        """
        Answer only if sufficient evidence is retrieved.
        
        Parameters
        ----------
        lambda_param : float
            Minimum number of supporting documents required
        query : str
            Input query
        retrieval_count : int, optional
            Number of retrieved supporting documents
        retrieved_docs : List[str], optional
            Retrieved documents
        **context
            Additional context
            
        Returns
        -------
        bool
            True if retrieval_count >= λ, False otherwise
        """
        if retrieval_count is not None:
            return retrieval_count >= lambda_param
        elif retrieved_docs is not None:
            return len(retrieved_docs) >= lambda_param
        else:
            # No retrieval info: abstain conservatively
            return False
            
    def get_response(
        self,
        lambda_param: float,
        query: str,
        base_response: str,
        **context
    ) -> str:
        """
        Return response with evidence if sufficient, otherwise abstain.
        """
        if self.should_answer(lambda_param, query, **context):
            # Optionally augment response with evidence
            retrieved_docs = context.get('retrieved_docs', [])
            if retrieved_docs:
                evidence_summary = f"\n\nBased on {len(retrieved_docs)} supporting sources."
                return base_response + evidence_summary
            return base_response
        else:
            return self.abstention_message


class ConsistencyPolicy(ControlPolicy):
    """
    Self-consistency policy requiring agreement across multiple samples.
    
    Only answers when λ or more independent samples agree.
    """
    
    def __init__(
        self,
        sampling_function: Optional[callable] = None,
        agreement_threshold: float = 0.8,
        abstention_message: str = "Insufficient consistency to answer."
    ):
        """
        Initialize consistency policy.
        
        Parameters
        ----------
        sampling_function : callable, optional
            Function to generate multiple samples
        agreement_threshold : float, default=0.8
            Threshold for considering samples as "agreeing"
        abstention_message : str
            Message when insufficient consistency
        """
        self.sampling_function = sampling_function
        self.agreement_threshold = agreement_threshold
        self.abstention_message = abstention_message
        
    def should_answer(
        self,
        lambda_param: float,
        query: str,
        consistency_count: Optional[int] = None,
        sample_responses: Optional[List[str]] = None,
        **context
    ) -> bool:
        """
        Answer only if sufficient samples agree.
        
        Parameters
        ----------
        lambda_param : float
            Minimum number of agreeing samples required
        query : str
            Input query
        consistency_count : int, optional
            Number of samples that agree with majority response
        sample_responses : List[str], optional
            Multiple sampled responses
        **context
            Additional context
            
        Returns
        -------
        bool
            True if consistency_count >= λ, False otherwise
        """
        if consistency_count is not None:
            return consistency_count >= lambda_param
        elif sample_responses is not None:
            # Compute agreement dynamically
            agreement_count = self._compute_agreement(sample_responses)
            return agreement_count >= lambda_param
        else:
            return False
            
    def get_response(
        self,
        lambda_param: float,
        query: str,
        base_response: str,
        **context
    ) -> str:
        """
        Return consensus response if sufficient agreement, otherwise abstain.
        """
        if self.should_answer(lambda_param, query, **context):
            # Could return majority/consensus response
            sample_responses = context.get('sample_responses', [base_response])
            if len(sample_responses) > 1:
                consensus_response = self._get_consensus(sample_responses)
                consistency_count = context.get('consistency_count', 1)
                return f"{consensus_response}\n\n(Consensus from {consistency_count} samples)"
            return base_response
        else:
            return self.abstention_message
            
    def _compute_agreement(self, responses: List[str]) -> int:
        """Compute number of responses agreeing with majority."""
        if not responses:
            return 0
            
        # Simple string similarity-based agreement
        from collections import Counter
        response_counts = Counter(responses)
        most_common_response, count = response_counts.most_common(1)[0]
        return count
        
    def _get_consensus(self, responses: List[str]) -> str:
        """Get consensus response from multiple samples."""
        from collections import Counter
        response_counts = Counter(responses)
        consensus_response, _ = response_counts.most_common(1)[0]
        return consensus_response


class MultiParameterPolicy(ControlPolicy):
    """
    Policy combining multiple control mechanisms.
    
    Uses a vector λ = (λ_conf, λ_retrieval, λ_consistency) to control
    multiple aspects simultaneously.
    """
    
    def __init__(
        self,
        confidence_policy: Optional[AbstractionPolicy] = None,
        retrieval_policy: Optional[RetrievalPolicy] = None,
        consistency_policy: Optional[ConsistencyPolicy] = None,
        combination_mode: str = "all"  # "all", "any", "majority"
    ):
        """
        Initialize multi-parameter policy.
        
        Parameters
        ----------
        confidence_policy : AbstractionPolicy, optional
            Confidence-based component
        retrieval_policy : RetrievalPolicy, optional
            Retrieval-based component
        consistency_policy : ConsistencyPolicy, optional
            Consistency-based component
        combination_mode : str, default="all"
            How to combine policy decisions:
            - "all": all policies must agree to answer
            - "any": any policy can approve answering
            - "majority": majority of policies must agree
        """
        self.confidence_policy = confidence_policy or AbstractionPolicy()
        self.retrieval_policy = retrieval_policy or RetrievalPolicy()
        self.consistency_policy = consistency_policy or ConsistencyPolicy()
        self.combination_mode = combination_mode
        
    def should_answer(
        self,
        lambda_param: Union[float, Tuple[float, ...], Dict[str, float]],
        query: str,
        **context
    ) -> bool:
        """
        Answer based on combination of multiple policies.
        
        Parameters
        ----------
        lambda_param : Union[float, Tuple, Dict]
            Control parameters. Can be:
            - float: use same value for all policies
            - tuple: (λ_conf, λ_retrieval, λ_consistency)
            - dict: {"confidence": λ_conf, "retrieval": λ_ret, "consistency": λ_cons}
        query : str
            Input query
        **context
            Context for all policies
            
        Returns
        -------
        bool
            Combined decision from all policies
        """
        # Parse lambda parameters
        if isinstance(lambda_param, (int, float)):
            lambda_conf = lambda_retrieval = lambda_consistency = lambda_param
        elif isinstance(lambda_param, (tuple, list)):
            lambda_conf, lambda_retrieval, lambda_consistency = lambda_param
        elif isinstance(lambda_param, dict):
            lambda_conf = lambda_param.get("confidence", 0.5)
            lambda_retrieval = lambda_param.get("retrieval", 1)
            lambda_consistency = lambda_param.get("consistency", 1)
        else:
            raise ValueError(f"Invalid lambda_param type: {type(lambda_param)}")
            
        # Get decisions from each policy
        decisions = []
        
        if self.confidence_policy:
            decisions.append(
                self.confidence_policy.should_answer(lambda_conf, query, **context)
            )
            
        if self.retrieval_policy:
            decisions.append(
                self.retrieval_policy.should_answer(lambda_retrieval, query, **context)
            )
            
        if self.consistency_policy:
            decisions.append(
                self.consistency_policy.should_answer(lambda_consistency, query, **context)
            )
            
        # Combine decisions
        if self.combination_mode == "all":
            return all(decisions)
        elif self.combination_mode == "any":
            return any(decisions)
        elif self.combination_mode == "majority":
            return sum(decisions) > len(decisions) / 2
        else:
            raise ValueError(f"Unknown combination mode: {self.combination_mode}")
            
    def get_response(
        self,
        lambda_param: Union[float, Tuple[float, ...], Dict[str, float]],
        query: str,
        base_response: str,
        **context
    ) -> str:
        """
        Generate response using combined policy logic.
        """
        if self.should_answer(lambda_param, query, **context):
            # Could enhance response with information from all policies
            enhancements = []
            
            # Add retrieval evidence if available
            retrieved_docs = context.get('retrieved_docs', [])
            if retrieved_docs:
                enhancements.append(f"Evidence from {len(retrieved_docs)} sources")
                
            # Add consistency information if available  
            consistency_count = context.get('consistency_count')
            if consistency_count and consistency_count > 1:
                enhancements.append(f"Confirmed by {consistency_count} samples")
                
            if enhancements:
                enhancement_text = " (" + ", ".join(enhancements) + ")"
                return base_response + enhancement_text
            return base_response
        else:
            return "I cannot provide a confident answer to this question."