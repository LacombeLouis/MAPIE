# Conformal Hallucination Mitigation (CHM)

This module implements a novel **Conformal Hallucination Mitigation (CHM)** framework that provides distribution-free theoretical guarantees for controlling LLM hallucination risk. The CHM system combines insights from conformal risk control with practical LLM deployment considerations.

## Overview

The CHM framework addresses a critical challenge in LLM deployment: **hallucinations are statistically inevitable** for well-calibrated models, but we can still provide **provable guarantees** on expected hallucination harm while maintaining utility.

### Key Features

- **Distribution-free guarantees**: E[hallucination_loss] ≤ α without assumptions on data distribution
- **Model-agnostic**: Works with any LLM via wrapper approach
- **Multiple control mechanisms**: Confidence thresholds, retrieval requirements, self-consistency
- **Risk-utility tradeoffs**: Simultaneous control of hallucination risk and answer coverage
- **Comprehensive benchmarking**: Standardized evaluation protocols and synthetic datasets

## Theoretical Foundation

CHM builds on two key papers:

1. **"Why Language Models Hallucinate"** - Shows hallucinations are statistically inevitable under calibration
2. **"Conformal Risk Control"** - Provides framework for controlling expected value of bounded monotone losses

The key insight: wrap LLMs with λ-dependent control policies and use conformal risk control to bound E[hallucination_harm] ≤ α.

## Architecture

```
Input Query → Base LLM → Control Policy(λ) → Controlled Response
                ↑
        Conformal Risk Control
        (calibrated λ* from training data)
```

### Core Components

1. **Loss Functions**: Quantify hallucination risk with monotonicity properties
2. **Control Policies**: Define λ-dependent LLM behavior (abstention, verification, etc.)
3. **CHM Controller**: Combines loss + policy with conformal risk control
4. **Benchmark Framework**: Evaluation and comparison tools

## Quick Start

```python
from mapie.hallucination import (
    HallucinationController,
    WeightedHallucinationLoss,
    AbstractionPolicy
)

# 1. Set up CHM components
loss_fn = WeightedHallucinationLoss(abstention_penalty=0.01)
policy = AbstractionPolicy()
controller = HallucinationController(loss_fn, policy)

# 2. Calibrate on validation data
controller.fit(
    calibration_queries=cal_queries,
    calibration_responses=cal_responses,
    calibration_ground_truth=cal_truth,
    alpha=0.05,  # Target 5% hallucination rate
    confidence_scores=cal_confidence
)

# 3. Use with guarantees
response = controller.predict(
    query="What is the capital of France?",
    base_response="Paris",
    confidence_score=0.95
)
# Guarantee: E[hallucination_loss] ≤ 0.05
```

## Components

### Loss Functions

#### `WeightedHallucinationLoss`
```python
loss = WeightedHallucinationLoss(
    weights=severity_weights,           # Per-sample severity
    abstention_penalty=0.01,            # Small abstention cost
    confidence_threshold_mode="confidence"  # or "retrieval"
)
```

Computes: `L_i(λ) = w_i * H_i(λ) + β * A_i(λ)`
- `H_i(λ)`: Hallucination indicator for sample i at threshold λ
- `A_i(λ)`: Abstention indicator 
- `w_i`: Severity weight
- `β`: Abstention penalty (β << min(w_i) for monotonicity)

#### `MultiRiskLoss`
```python
multi_loss = MultiRiskLoss(
    hallucination_loss=base_loss,
    utility_weight=1.0
)
```

Enables simultaneous control of:
- Expected hallucination harm ≤ α_h
- Expected abstention rate ≤ α_u

### Control Policies

#### `AbstractionPolicy`
Confidence-based abstention:
```python
policy = AbstractionPolicy(abstention_message="I don't know.")
```

#### `RetrievalPolicy`
Evidence-based control:
```python
policy = RetrievalPolicy(
    retrieval_function=retrieve_docs,
    abstention_message="Insufficient evidence."
)
```

#### `ConsistencyPolicy`
Self-consistency based control:
```python
policy = ConsistencyPolicy(
    sampling_function=sample_responses,
    agreement_threshold=0.8
)
```

#### `MultiParameterPolicy`
Combines multiple mechanisms:
```python
policy = MultiParameterPolicy(
    confidence_policy=AbstractionPolicy(),
    retrieval_policy=RetrievalPolicy(),
    consistency_policy=ConsistencyPolicy(),
    combination_mode="all"  # "all", "any", "majority"
)
```

### Benchmark Framework

#### `HallucinationBenchmark`
```python
benchmark = HallucinationBenchmark(
    dataset=SyntheticQADataset(n_samples=1000),
    llm_simulator=MockLLMSimulator(hallucination_rate=0.15)
)

# Evaluate single controller
results = benchmark.evaluate_controller(controller, alpha_values=[0.05, 0.1])

# Compare multiple controllers
comparison = benchmark.compare_controllers({
    "conservative": controller1,
    "aggressive": controller2
})

# Generate report
report = benchmark.generate_report(results)
```

## Advanced Usage

### Multi-Risk Control
```python
# Control both hallucination risk and utility
controller.fit(..., alpha=0.1)  # 10% hallucination risk

# Alternative: control multiple risks simultaneously
alpha_dict = {"hallucination": 0.05, "utility": 0.2}
# (Would require extended multi-risk implementation)
```

### Custom Loss Functions
```python
class CustomHallucinationLoss(HallucinationLoss):
    def __call__(self, responses, ground_truth, lambda_param, **kwargs):
        # Custom logic ensuring monotonicity in lambda_param
        return loss_values
```

### Real LLM Integration
```python
# Wrapper for real LLM
class LLMWrapper:
    def __init__(self, model):
        self.model = model
        
    def generate_with_confidence(self, query):
        response = self.model.generate(query)
        confidence = self.model.get_confidence()  # Model-specific
        return response, confidence

# Use with CHM
llm = LLMWrapper(your_model)
response, confidence = llm.generate_with_confidence(query)
controlled_response = controller.predict(query, response, confidence_score=confidence)
```

## Theoretical Guarantees

For any fitted CHM controller:

**Main Guarantee**: E[L(Y_{n+1}, λ̂)] ≤ α + O(1/n)

Where:
- L: Chosen loss function  
- Y_{n+1}: Next sample
- λ̂: Calibrated threshold
- α: Target risk level
- O(1/n): Finite-sample slack

**Key Properties**:
- Distribution-free (no assumptions on data/model)
- Finite-sample guarantees (not asymptotic)
- Model-agnostic (works with any LLM)
- Monotone loss requirement ensures guarantees hold

## Examples

See `examples/hallucination_example.py` for comprehensive demonstrations including:

1. **Basic CHM**: Simple confidence-based control
2. **Benchmark Comparison**: Comparing different configurations  
3. **Multi-Risk Control**: Hallucination vs. utility tradeoffs

## Integration with MAPIE

CHM extends MAPIE's conformal prediction capabilities:

- **Regression/Classification**: Prediction intervals/sets for traditional ML
- **Risk Control**: General risk control for complex metrics
- **Hallucination Control**: Specialized for LLM hallucination mitigation

## Research Applications

The CHM framework enables research in:

1. **Benchmark Development**: Standardized hallucination evaluation
2. **Policy Design**: Novel λ-dependent control mechanisms  
3. **Loss Function Innovation**: New monotone hallucination losses
4. **Multi-Objective Control**: Complex risk-utility tradeoffs
5. **Real-World Deployment**: Production LLM safety systems

## Citation

If you use CHM in your research, please cite:

```bibtex
@article{mapie_chm_2024,
  title={Conformal Hallucination Mitigation for Large Language Models},
  author={MAPIE Contributors},
  journal={MAPIE Library},
  year={2024},
  note={Implementation of conformal risk control for LLM hallucination mitigation}
}
```

## References

1. Kalai, A., Nachum, O., Vempala, S., Zhang, D. (2025). "Why Language Models Hallucinate"
2. Angelopoulos, A.N., Bates, S., Fisch, A., Lei, L., Schuster, T. (2025). "Conformal Risk Control"
3. Bates, S., et al. (2021). "Distribution-free, risk-controlling prediction sets"
4. Romano, Y., et al. (2020). "Classification with valid and adaptive coverage"