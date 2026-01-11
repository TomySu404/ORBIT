"""
ORBIT: On-distribution Rollout-based Behavioral Intervention Technique.

Key improvements:
1. On-distribution Rollout Contrast (ORC)
2. Continuous Soft Scaling (CSS)
3. Structural Layer-wise Ablation
"""

__version__ = "0.1.0"
__author__ = "Your Name"

from .config import ExperimentConfig, RolloutConfig, InterventionConfig, LayerScope, ModelType
