"""
Continuous difference vector computation with soft scaling.

This module implements the second key improvement:
Instead of Top-K selection + binary masking, we use continuous scaling weights.

Mathematical motivation:
Let v* be the ideal steering direction (true semantic difference).
Discrete approach: v_discrete = mask ⊙ v*, where mask ∈ {0,1}^d
Our method: v_ours = β ⊙ v*, where β ∈ [-1,1]^d (continuous)

The cosine similarity between v_discrete and v* is strictly less than 1
whenever v* has non-zero components in masked dimensions.
Our continuous scaling preserves the direction: cos(v_ours, v*) ≈ 1.
"""
import torch
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass, field

from ..config import InterventionConfig, LayerScope


@dataclass
class DiffVectorResult:
    """
    Result of difference vector computation.
    
    Attributes:
        diff_vectors: Mean difference vectors per layer.
        scaling_weights: Continuous scaling weights (beta) per layer.
        sample_count: Total number of samples used.
        pair_count: Total number of contrastive pairs.
        reread_sample_count: Number of samples using re-read mechanism.
    """
    diff_vectors: Dict[str, torch.Tensor] = field(default_factory=dict)
    scaling_weights: Dict[str, torch.Tensor] = field(default_factory=dict)
    sample_count: int = 0
    pair_count: int = 0
    reread_sample_count: int = 0


class ContinuousDiffCalculator:
    """
    Computes continuous difference vectors with soft scaling.
    
    Key improvements:
    1. No Top-K selection - preserves ALL dimensions
    2. Continuous beta weights based on magnitude
    3. Multiple scaling methods for flexibility
    4. Layer-wise computation for ablation studies
    
    Scaling methods:
    - max_norm: β_i = Δh_i / max(|Δh|)  [preserves relative magnitude]
    - l2_norm: β_i = Δh_i / ||Δh||_2    [unit vector]
    - softmax: β_i = softmax(|Δh|)_i * sign(Δh_i) * d  [emphasizes large diffs]
    """
    
    def __init__(self, model, config: InterventionConfig):
        """
        Initialize difference calculator.
        
        Args:
            model: ModelWrapper instance.
            config: Intervention configuration.
        """
        self.model = model
        self.config = config
        self._layer_indices = self._compute_layer_indices()
    
    def _compute_layer_indices(self) -> List[int]:
        """
        Determine which layers to intervene on based on configuration.
        
        Supports:
        - ALL: All layers
        - FIRST_N: First N layers (shallow)
        - LAST_N: Last N layers (deep)
        - CUSTOM: User-specified layer indices
        
        Returns:
            List of layer indices.
        """
        total_layers = self.model.num_layers
        scope = self.config.layer_scope
        n = self.config.num_layers
        
        if scope == LayerScope.ALL:
            return list(range(total_layers))
        
        elif scope == LayerScope.FIRST_N:
            return list(range(min(n, total_layers)))
        
        elif scope == LayerScope.LAST_N:
            start = max(0, total_layers - n)
            return list(range(start, total_layers))
        
        elif scope == LayerScope.CUSTOM:
            if self.config.custom_layers:
                return [i for i in self.config.custom_layers if 0 <= i < total_layers]
            return list(range(total_layers))
        
        return list(range(total_layers))
    
    @property
    def layer_indices(self) -> List[int]:
        """Get the layer indices being used for intervention."""
        return self._layer_indices
    
    def _compute_scaling_weights(
        self,
        diff: torch.Tensor,
        method: str
    ) -> torch.Tensor:
        """
        Compute continuous scaling weights (beta) from difference vector.
        
        This is the core innovation replacing binary masking.
        
        Mathematical formulation:
        - max_norm: β = Δh / max(|Δh|), β ∈ [-1, 1]^d
        - l2_norm: β = Δh / ||Δh||_2, ||β||_2 = 1
        - softmax: β = d * softmax(|Δh|) * sign(Δh)
        
        Args:
            diff: Difference vector [hidden_dim].
            method: Scaling method name.
        
        Returns:
            Scaling weights tensor [hidden_dim].
        """
        eps = 1e-8  # Numerical stability
        
        if method == "max_norm":
            # Scale by maximum absolute value
            # Preserves relative magnitudes, β ∈ [-1, 1]
            max_val = torch.max(torch.abs(diff))
            if max_val > eps:
                return diff / max_val
            return torch.zeros_like(diff)
        
        elif method == "l2_norm":
            # L2 normalize to unit vector
            # Direction preserved exactly
            norm = torch.norm(diff, p=2)
            if norm > eps:
                return diff / norm
            return torch.zeros_like(diff)
        
        elif method == "softmax":
            # Softmax over absolute values
            # Emphasizes dimensions with larger differences
            abs_diff = torch.abs(diff)
            weights = torch.softmax(abs_diff, dim=-1)
            # Scale by dimension and restore sign
            return weights * torch.sign(diff) * diff.shape[-1]
        
        else:
            raise ValueError(f"Unknown scaling method: {method}")
    
    def compute_pair_diff(
        self,
        question: str,
        positive_answer: str,
        negative_answer: str,
        components: Optional[List[str]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute activation difference for a single contrastive pair.
        
        Extracts activations at the last token position for both
        (question + positive) and (question + negative), then computes
        the element-wise difference.
        
        Args:
            question: Input question/prompt.
            positive_answer: Correct answer.
            negative_answer: Incorrect answer.
            components: Activation components to extract.
        
        Returns:
            Dictionary mapping layer names to difference tensors.
        """
        if components is None:
            components = self.config.components
        
        # Construct full input strings
        pos_text = f"{question}{positive_answer}"
        neg_text = f"{question}{negative_answer}"
        
        # Get activations for positive (correct) input
        pos_activations = self.model.get_activations(
            pos_text,
            layer_indices=self._layer_indices,
            components=components,
            token_position=-1  # Last token
        )
        
        # Get activations for negative (incorrect) input
        neg_activations = self.model.get_activations(
            neg_text,
            layer_indices=self._layer_indices,
            components=components,
            token_position=-1
        )
        
        # Compute element-wise difference: positive - negative
        diff = {}
        for name in pos_activations:
            diff[name] = pos_activations[name] - neg_activations[name]
        
        return diff
    
    def aggregate_diffs(
        self,
        all_diffs: List[Dict[str, torch.Tensor]],
        reread_flags: Optional[List[bool]] = None,
        reread_weight: float = 0.5
    ) -> DiffVectorResult:
        """
        Aggregate difference vectors from multiple pairs.
        
        Implements improved averaging strategy:
        1. Weighted sum across all pairs
        2. Down-weight re-read (forced) samples
        3. Compute mean difference vector
        4. Apply continuous scaling
        
        Mathematical formulation:
        Let D = {(Δh_i, w_i)} be the set of difference vectors with weights.
        Mean: μ = Σ(w_i * Δh_i) / Σ(w_i)
        Scaling: β = scaling_method(μ)
        
        The Law of Large Numbers ensures that as we aggregate more pairs,
        sample-specific noise cancels out, leaving the true semantic direction.
        
        Args:
            all_diffs: List of diff dictionaries from compute_pair_diff.
            reread_flags: Optional flags indicating re-read pairs.
            reread_weight: Weight multiplier for re-read pairs (< 1 to down-weight).
        
        Returns:
            DiffVectorResult with aggregated vectors and scaling weights.
        """
        if not all_diffs:
            raise ValueError("No difference vectors to aggregate")
        
        # Initialize accumulators
        layer_names = list(all_diffs[0].keys())
        accumulated = {name: torch.zeros_like(all_diffs[0][name]) for name in layer_names}
        total_weight = 0.0
        reread_count = 0
        
        # Weighted accumulation
        for i, diff in enumerate(all_diffs):
            # Determine weight for this pair
            if reread_flags and i < len(reread_flags) and reread_flags[i]:
                weight = reread_weight
                reread_count += 1
            else:
                weight = 1.0
            
            # Accumulate weighted difference
            for name in layer_names:
                accumulated[name] += weight * diff[name]
            total_weight += weight
        
        # Compute weighted mean
        if total_weight > 0:
            mean_diff = {
                name: accumulated[name] / total_weight
                for name in layer_names
            }
        else:
            mean_diff = accumulated
        
        # Compute continuous scaling weights using configured method
        scaling_weights = {
            name: self._compute_scaling_weights(
                mean_diff[name],
                self.config.scaling_method
            )
            for name in layer_names
        }
        
        return DiffVectorResult(
            diff_vectors=mean_diff,
            scaling_weights=scaling_weights,
            sample_count=len(all_diffs),
            pair_count=len(all_diffs),
            reread_sample_count=reread_count
        )
