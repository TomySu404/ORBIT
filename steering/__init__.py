"""
Steering module for improved activation intervention.

Contains:
- RolloutGenerator: Generate contrastive pairs via model rollout
- ContinuousDiffCalculator: Compute continuous difference vectors
- ActivationIntervention: Apply steering during inference
"""

from .rollout import RolloutGenerator, RolloutResult, ContrastivePair
from .diff_vector import ContinuousDiffCalculator, DiffVectorResult
from .intervention import ActivationIntervention

__all__ = [
    "RolloutGenerator",
    "RolloutResult", 
    "ContrastivePair",
    "ContinuousDiffCalculator",
    "DiffVectorResult",
    "ActivationIntervention",
]
