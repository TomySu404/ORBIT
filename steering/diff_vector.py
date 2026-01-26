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
from typing import Dict, List, Optional
from dataclasses import dataclass, field

from config import InterventionConfig, LayerScope


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
    4. Layer-wise computation
    
    Scaling methods:
    - max_norm: β_i = Δh_i / max(|Δh|)  [preserves relative magnitude]
    - l2_norm: β_i = Δh_i / ||Δh||_2    [unit vector]
    - softmax: β_i = softmax(|Δh|)_i * sign(Δh_i) * d  [emphasizes large diffs]
    """
    
    def __init__(self, model, config: InterventionConfig, format_type: str = "generation", enable_thinking: bool = False):
        """
        Initialize difference calculator.

        Args:
            model: ModelWrapper instance.
            config: Intervention configuration.
            format_type: Format type for prompt construction ("generation" or "chat").
            enable_thinking: Whether to enable thinking tokens in chat template.
        """
        self.model = model
        self.config = config
        self.format_type = format_type
        self.enable_thinking = enable_thinking
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
        
        Uses batch inference to process positive and negative together.
        
        Args:
            question: Input question/prompt.
            positive_answer: Correct answer.
            negative_answer: Incorrect answer.
            components: Activation components to extract.
        
        Returns:
            Dictionary mapping layer names to difference tensors [hidden_dim].
        """
        if components is None:
            components = self.config.components
        
        # Construct full input strings using format_type
        pos_text = self.model.format_prompt(question, positive_answer, self.format_type, self.enable_thinking)
        neg_text = self.model.format_prompt(question, negative_answer, self.format_type, self.enable_thinking)
        
        # Batch inference: process positive and negative together
        activations = self.model.get_activations(
            [pos_text, neg_text],  # Batch of 2
            layer_indices=self._layer_indices,
            components=components,
            token_position=self.config.steering_token_position
        )
        
        # Compute element-wise difference: positive - negative
        # activations[name] shape: [2, hidden_dim]
        diff = {}
        for name in activations:
            diff[name] = activations[name][0] - activations[name][1]
        
        return diff
    
    def compute_batch_pair_diffs(
        self,
        questions: List[str],
        positive_answers: List[str],
        negative_answers: List[str],
        components: Optional[List[str]] = None
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Compute activation differences for multiple contrastive pairs in batch.
        
        This method processes all pairs in a single forward pass for efficiency.
        
        Args:
            questions: List of input questions/prompts.
            positive_answers: List of correct answers.
            negative_answers: List of incorrect answers.
            components: Activation components to extract.
        
        Returns:
            List of dictionaries, each mapping layer names to diff tensors.
        """
        if components is None:
            components = self.config.components
        
        n_pairs = len(questions)
        if n_pairs == 0:
            return []
        
        # Construct all input texts: [pos1, neg1, pos2, neg2, ...] using format_type
        all_texts = []
        for q, pos, neg in zip(questions, positive_answers, negative_answers):
            all_texts.append(self.model.format_prompt(q, pos, self.format_type, self.enable_thinking))
            all_texts.append(self.model.format_prompt(q, neg, self.format_type, self.enable_thinking))
        
        # Single batch forward pass
        activations = self.model.get_activations(
            all_texts,
            layer_indices=self._layer_indices,
            components=components,
            token_position=self.config.steering_token_position
        )
        
        # Extract diffs for each pair
        # activations[name] shape: [2*n_pairs, hidden_dim]
        all_diffs = []
        for i in range(n_pairs):
            pos_idx = 2 * i
            neg_idx = 2 * i + 1
            diff = {}
            for name in activations:
                diff[name] = activations[name][pos_idx] - activations[name][neg_idx]
            all_diffs.append(diff)
        
        return all_diffs
    
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
        n_pairs = len(all_diffs)
        first_tensor = all_diffs[0][layer_names[0]]
        device = first_tensor.device
        dtype = first_tensor.dtype
        
        # Pre-compute weights as a tensor for vectorized operations
        if reread_flags and len(reread_flags) >= n_pairs:
            weights = torch.tensor([
                reread_weight if flag else 1.0 
                for flag in reread_flags[:n_pairs]
            ], device=device, dtype=dtype)
            reread_count = sum(reread_flags[:n_pairs])
        else:
            weights = torch.ones(n_pairs, device=device, dtype=dtype)
            reread_count = 0
            
        total_weight = weights.sum().item()
        
        # Vectorized aggregation per layer
        # This replaces the slow nested Python loop with efficient torch operations
        mean_diff = {}
        for name in layer_names:
            # Stack all diffs for this layer: [n_pairs, hidden_dim]
            layer_tensors = torch.stack([d[name] for d in all_diffs])
            
            # Weighted sum: [1, n_pairs] @ [n_pairs, hidden_dim] -> [1, hidden_dim]
            if total_weight > 0:
                weighted_sum = (weights.unsqueeze(0) @ layer_tensors).squeeze(0)
                mean_diff[name] = weighted_sum / total_weight
            else:
                mean_diff[name] = torch.zeros_like(layer_tensors[0])
        
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
    
    def aggregate_diffs_grouped(
        self,
        all_diffs: List[Dict[str, torch.Tensor]],
        pairs_per_question: List[int],
        reread_flags: Optional[List[bool]] = None,
        reread_weight: float = 0.5
    ) -> DiffVectorResult:
        """
        Aggregate difference vectors using per-question normalization before global averaging.
        
        This implements a "One Question, One Vote" strategy:
        1. For each question, compute weighted average of its pairs
        2. Normalize each question's vector (using configured scaling method)
        3. Average the normalized vectors across all questions
        
        This approach:
        - Gives equal contribution to each question regardless of pair count
        - Prevents high-activation samples from dominating the final vector
        - Better handles datasets with varying sample difficulty/length
        
        Args:
            all_diffs: List of diff dictionaries from compute_pair_diff.
            pairs_per_question: List of pair counts per question (sum = len(all_diffs)).
            reread_flags: Optional flags indicating re-read pairs.
            reread_weight: Weight multiplier for re-read pairs.
        
        Returns:
            DiffVectorResult with grouped-normalized aggregated vectors.
        """
        if not all_diffs:
            raise ValueError("No difference vectors to aggregate")
        
        if not pairs_per_question or sum(pairs_per_question) != len(all_diffs):
            # Fallback to standard aggregation if grouping info is invalid
            return self.aggregate_diffs(all_diffs, reread_flags, reread_weight)
        
        layer_names = list(all_diffs[0].keys())
        first_tensor = all_diffs[0][layer_names[0]]
        device = first_tensor.device
        dtype = first_tensor.dtype
        
        # Store normalized vectors for each question
        question_vectors = {name: [] for name in layer_names}
        total_reread_count = 0
        
        current_idx = 0
        for num_pairs in pairs_per_question:
            if num_pairs <= 0:
                continue
                
            # 1. Extract diffs belonging to this question
            q_diffs = all_diffs[current_idx : current_idx + num_pairs]
            q_flags = reread_flags[current_idx : current_idx + num_pairs] if reread_flags else None
            
            # 2. Compute intra-group weighted average
            q_result = self.aggregate_diffs(q_diffs, q_flags, reread_weight)
            total_reread_count += q_result.reread_sample_count
            
            # 3. Store the normalized vector (scaling_weights already computed in aggregate_diffs)
            # The "steering direction" for this question is: diff_vector (already mean)
            # We then apply the scaling to get a normalized direction
            for name in layer_names:
                beta_q = q_result.scaling_weights[name]
                # Normalized direction vector for this question
                question_vectors[name].append(beta_q)
            
            current_idx += num_pairs
        
        # 4. Global average across all questions (equal weight per question)
        final_diff_vectors = {}
        for name in layer_names:
            if question_vectors[name]:
                stacked = torch.stack(question_vectors[name])
                final_diff_vectors[name] = stacked.mean(dim=0)
            else:
                final_diff_vectors[name] = torch.zeros_like(first_tensor)
        
        # 5. Since we already applied scaling per-question, set final scaling_weights to ones
        # This ensures that during intervention: h' = h + alpha * (1.0) * final_diff
        # The direction is already "baked in" from per-question normalization
        final_scaling_weights = {
            name: torch.ones_like(v) for name, v in final_diff_vectors.items()
        }
        
        return DiffVectorResult(
            diff_vectors=final_diff_vectors,
            scaling_weights=final_scaling_weights,
            sample_count=len(pairs_per_question),
            pair_count=len(all_diffs),
            reread_sample_count=total_reread_count
        )