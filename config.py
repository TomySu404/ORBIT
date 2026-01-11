"""
Configuration classes for the ORBIT method.

This module defines all configuration dataclasses used throughout the project,
providing a clean and type-safe way to manage hyperparameters.
"""
from dataclasses import dataclass, field
from typing import List, Literal, Optional
from enum import Enum


class ModelType(Enum):
    """Supported model architectures."""
    LLAMA3 = "llama3"
    QWEN3 = "qwen3"
    GLM = "glm"
    GEMMA3 = "gemma3"


class LayerScope(Enum):
    """
    Layer intervention scope for ablation study.
    
    - FIRST_N: Intervene on first N layers (shallow layers)
    - LAST_N: Intervene on last N layers (deep layers)
    - ALL: Intervene on all layers
    - CUSTOM: Intervene on custom-specified layers
    """
    FIRST_N = "first_n"
    LAST_N = "last_n"
    ALL = "all"
    CUSTOM = "custom"


@dataclass
class RolloutConfig:
    """
    Configuration for rollout-based response generation.
    
    Attributes:
        num_rollouts: Number of diverse responses to generate per sample.
        temperature: Sampling temperature for diversity.
        top_p: Nucleus sampling probability threshold.
        max_new_tokens: Maximum tokens to generate per response.
        use_reread_fallback: Whether to use re-read mechanism when no correct rollout found.
    """
    num_rollouts: int = 8
    temperature: float = 0.8
    top_p: float = 0.9
    max_new_tokens: int = 32
    use_reread_fallback: bool = True


@dataclass
class InterventionConfig:
    """
    Configuration for activation intervention.
    
    Attributes:
        layer_scope: Which layers to intervene on (for ablation study).
        num_layers: Number of layers for FIRST_N or LAST_N scope.
        custom_layers: Specific layer indices for CUSTOM scope.
        scaling_method: Method for computing continuous scaling weights.
            - 'softmax': Softmax over absolute differences
            - 'l2_norm': L2 normalized difference vector
            - 'max_norm': Scale by maximum absolute value
        intervention_strength: Global scaling factor for intervention.
        components: Model components to intervene on.
    """
    layer_scope: LayerScope = LayerScope.ALL
    num_layers: int = 5
    custom_layers: Optional[List[int]] = None
    scaling_method: Literal["softmax", "l2_norm", "max_norm"] = "max_norm"
    intervention_strength: float = 1.0
    components: List[str] = field(default_factory=lambda: ["mlp_act"])


@dataclass
class ExperimentConfig:
    """
    Main experiment configuration.
    
    Attributes:
        model_name: HuggingFace model identifier.
        model_type: Model architecture type (auto-detected if not specified).
        device: Computation device ('cuda' or 'cpu').
        dtype: Model precision ('float16', 'bfloat16', 'float32').
        rollout: Rollout generation configuration.
        intervention: Intervention configuration.
        train_samples: Maximum number of training samples for building steering vectors.
        reread_weight: Weight multiplier for re-read samples (down-weight if < 1.0).
        seed: Random seed for reproducibility.
    """
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    model_type: Optional[ModelType] = None
    device: str = "cuda"
    dtype: str = "float16"
    
    rollout: RolloutConfig = field(default_factory=RolloutConfig)
    intervention: InterventionConfig = field(default_factory=InterventionConfig)
    
    train_samples: int = 150
    reread_weight: float = 0.5
    seed: int = 42
    
    def __post_init__(self):
        """Auto-detect model type from model name if not specified."""
        if self.model_type is None:
            name_lower = self.model_name.lower()
            if "llama" in name_lower:
                self.model_type = ModelType.LLAMA3
            elif "qwen" in name_lower:
                self.model_type = ModelType.QWEN3
            elif "glm" in name_lower:
                self.model_type = ModelType.GLM
            elif "gemma" in name_lower:
                self.model_type = ModelType.GEMMA3
            else:
                self.model_type = ModelType.LLAMA3  # Default fallback
