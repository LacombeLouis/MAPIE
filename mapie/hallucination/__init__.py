"""
Conformal Hallucination Mitigation (CHM) module.

This module implements distribution-free conformal risk control for LLM hallucination
mitigation, providing theoretical guarantees on expected hallucination harm while
maintaining utility.
"""

from .controller import HallucinationController
from .losses import (
    HallucinationLoss,
    WeightedHallucinationLoss,
    MultiRiskLoss
)
from .policies import (
    AbstractionPolicy,
    RetrievalPolicy,
    ConsistencyPolicy,
    MultiParameterPolicy
)
from .benchmark import HallucinationBenchmark

__all__ = [
    "HallucinationController",
    "HallucinationLoss",
    "WeightedHallucinationLoss", 
    "MultiRiskLoss",
    "AbstractionPolicy",
    "RetrievalPolicy",
    "ConsistencyPolicy",
    "MultiParameterPolicy",
    "HallucinationBenchmark"
]