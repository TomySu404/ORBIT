"""
Configuration module for ORBIT.

Defines enums and dataclasses for experiment settings, model architectures,
rollout parameters, and intervention strategies.
"""
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Optional, Any


class ModelType(Enum):
    """Supported model architectures."""
    LLAMA3 = "llama3"
    QWEN3 = "qwen3"
    GLM = "glm"
    GEMMA3 = "gemma3"


class LayerScope(Enum):
    """Scope of layers to apply intervention on."""
    ALL = "all"
    FIRST_N = "first_n"
    LAST_N = "last_n"
    CUSTOM = "custom"


@dataclass
class RolloutConfig:
    """Configuration for rollout generation (ORC)."""
    num_rollouts: int = 8
    temperature: float = 0.8
    top_p: float = 0.9
    max_new_tokens: int = 4
    use_reread_fallback: bool = True


@dataclass
class InterventionConfig:
    """Configuration for activation intervention (CSS)."""
    layer_scope: LayerScope = LayerScope.FIRST_N
    num_layers: int = 5
    scaling_method: str = "max_norm"  # choices: "softmax", "l2_norm", "max_norm"
    intervention_strength: float = 1.0
    components: List[str] = field(default_factory=lambda: ["mlp_act"])
    prefill_only: bool = True
    custom_layers: Optional[List[int]] = None


@dataclass
class ExperimentConfig:
    """Global experiment configuration."""
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    dtype: str = "bfloat16"
    device: str = "cuda"
    rollout: RolloutConfig = field(default_factory=RolloutConfig)
    intervention: InterventionConfig = field(
        default_factory=InterventionConfig
    )
    train_samples: Optional[int] = None
    reread_weight: float = 0.5
    seed: int = 42
    model_type: Optional[ModelType] = None

