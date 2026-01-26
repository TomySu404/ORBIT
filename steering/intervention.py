"""
Layer-wise activation intervention during inference.

This module applies the computed steering vectors to model activations,
implementing the third key improvement: continuous scaling intervention.

Intervention formula:
h' = h + α * β * μ

Where:
- h: Original activation
- α: Intervention strength (hyperparameter)
- β: Continuous scaling weights (from diff_vector.py)
- μ: Mean difference vector (from diff_vector.py)
"""
import torch
from torch import nn
from typing import Dict, List, Callable, Union
from contextlib import contextmanager

from config import InterventionConfig
from .diff_vector import DiffVectorResult


class ActivationIntervention:
    """
    Applies steering vectors to model activations during inference.
    
    Features:
    1. Continuous scaling (not binary masking)
    2. Layer-wise control
    3. Adjustable intervention strength
    4. Efficient hook-based implementation
    
    Mathematical note:
    The intervention h' = h + α*β*μ can be viewed as moving the activation
    along the direction that maximally differentiates correct from incorrect
    responses, with magnitude controlled by α and β.
    """
    
    def __init__(
        self,
        model,
        config: InterventionConfig,
        diff_result: DiffVectorResult
    ):
        """
        Initialize intervention handler.
        
        Args:
            model: ModelWrapper instance.
            config: Intervention configuration.
            diff_result: Pre-computed difference vectors and scaling weights.
        """
        self.model = model
        self.config = config
        self.diff_result = diff_result
        
        # Pre-compute intervention vectors for efficiency
        self._intervention_vectors = self._precompute_interventions()
        
        # Track initial sequence length for prefill-only intervention
        self._initial_seq_len = None
        # Mutable reference for hook functions (needed for pickle compatibility)
        self._initial_seq_len_ref = [None]
    
    def _precompute_interventions(self) -> Dict[str, torch.Tensor]:
        """
        Pre-compute intervention vectors: α * β * μ
        
        This is done once during initialization for efficiency,
        as the same vectors are applied across all inference calls.
        
        Returns:
            Dictionary mapping layer names to intervention tensors.
        """
        interventions = {}
        strength = self.config.intervention_strength
        
        for name in self.diff_result.diff_vectors:
            beta = self.diff_result.scaling_weights[name]
            
            # Intervention = strength * beta
            # Note: beta is already computed from diff_vectors (mu) with scaling
            # For max_norm scaling, beta = mu/max(|mu|)
            # So intervention = strength * mu / max(|mu|)
            # This naturally emphasizes dimensions with larger differences
            intervention = strength * beta
            # Move to model device
            interventions[name] = intervention.to(self.model.device)
        
        return interventions
    
    def _get_module(self, name: str) -> nn.Module:
        """
        Get a module by its dot-separated path.
        
        Args:
            name: Module path (e.g., 'model.layers.0.mlp.act_fn').
        
        Returns:
            The requested nn.Module.
        """
        module = self.model.model
        for part in name.split('.'):
            if part.isdigit():
                module = module[int(part)]
            else:
                module = getattr(module, part)
        return module
    
    @staticmethod
    def _intervention_hook(
        module: nn.Module,
        input,
        output,
        intervention: torch.Tensor,
        config_prefill_only: bool,
        initial_seq_len_ref: List,  # Use list to allow mutation
        token_position: int = -1
    ):
        """
        Static intervention hook that can be pickled.

        Args:
            module: The hooked module
            input: Input to the module
            output: Output from the module
            intervention: Pre-computed intervention tensor
            config_prefill_only: Whether to only intervene during prefill
            initial_seq_len_ref: Mutable reference to initial sequence length [seq_len]
            token_position: Token position to intervene
        """
        # Handle tuple outputs (some layers return (hidden_states, ...))
        if isinstance(output, tuple):
            hidden = output[0]
            rest = output[1:]
            is_tuple = True
        else:
            hidden = output
            rest = None
            is_tuple = False

        # Check if prefill-only intervention is enabled
        current_seq_len = hidden.shape[1]
        if config_prefill_only:
            # Initialize sequence length on first forward pass
            if initial_seq_len_ref[0] is None:
                initial_seq_len_ref[0] = current_seq_len

            # Only intervene if sequence length equals initial length (prefill phase)
            # Once sequence length increases, we're in generation phase - skip intervention
            if current_seq_len != initial_seq_len_ref[0]:
                # Generation phase - skip intervention
                if is_tuple:
                    return (hidden,) + rest
                return hidden

        # Ensure intervention is on correct device and dtype
        interv = intervention.to(hidden.device).to(hidden.dtype)

        # Apply intervention at specified token position
        # hidden shape: [batch, seq_len, hidden_dim]
        if token_position == -1:
            # Intervene on last token (most common case)
            hidden = hidden.clone()
            hidden[:, -1, :] = hidden[:, -1, :] + interv
        elif token_position >= 0:
            hidden = hidden.clone()
            hidden[:, token_position, :] = hidden[:, token_position, :] + interv
        else:
            # Negative indexing
            hidden = hidden.clone()
            hidden[:, token_position, :] = hidden[:, token_position, :] + interv

        # Reconstruct output
        if is_tuple:
            return (hidden,) + rest
        return hidden

    def _create_intervention_hook(
        self,
        layer_name: str,
        token_position: int = -1
    ) -> Callable:
        """
        Create a forward hook that adds intervention to activations.

        The hook modifies activations in-place at the specified token position.

        Args:
            layer_name: Name of the layer to intervene.
            token_position: Token position to intervene (-1 for last).

        Returns:
            Hook function compatible with PyTorch's register_forward_hook.
        """
        intervention = self._intervention_vectors[layer_name]

        # Use functools.partial to bind parameters without creating a closure
        from functools import partial

        # Sync the reference with current state
        self._initial_seq_len_ref[0] = self._initial_seq_len

        return partial(
            self._intervention_hook,
            intervention=intervention,
            config_prefill_only=self.config.prefill_only,
            initial_seq_len_ref=self._initial_seq_len_ref,
            token_position=token_position
        )
    
    @contextmanager
    def intervene(self, token_position: int = -1):
        """
        Context manager to apply intervention during forward pass.
        
        Registers hooks on all relevant layers, yields control,
        then cleans up hooks.
        
        Usage:
            with intervention.intervene():
                output = model.generate(prompt)
        
        Args:
            token_position: Token position to intervene (-1 for last).
        
        Yields:
            None (intervention is applied via hooks).
        """
        # Reset sequence length tracking for prefill-only mode
        if self.config.prefill_only:
            self._initial_seq_len = None
            self._initial_seq_len_ref[0] = None
        
        handles = []
        
        try:
            # Register intervention hooks for all layers
            for layer_name in self._intervention_vectors:
                try:
                    module = self._get_module(layer_name)
                    hook = self._create_intervention_hook(layer_name, token_position)
                    handle = module.register_forward_hook(hook)
                    handles.append(handle)
                except (AttributeError, KeyError) as e:
                    print(f"Warning: Could not hook {layer_name}: {e}")
            
            yield
            
        finally:
            # Clean up all hooks
            for handle in handles:
                handle.remove()
            # Reset sequence length tracking
            if self.config.prefill_only:
                self._initial_seq_len = None
                self._initial_seq_len_ref[0] = None
    
    @torch.no_grad()
    def generate_with_intervention(
        self,
        prompt: Union[str, List[str]],
        max_new_tokens: int = 32,
        temperature: float = 1.0,
        do_sample: bool = False,
        token_position: int = -1
    ) -> Union[str, List[str]]:
        """
        Generate response with activation intervention applied.
        
        The intervention is applied at the specified token position
        during each forward pass of the generation loop.
        
        Args:
            prompt: Input prompt or list of prompts.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            do_sample: Whether to use sampling.
            token_position: Token position to intervene (-1 for last).
        
        Returns:
            Generated response string or list of response strings.
        """
        with self.intervene(token_position=token_position):
            responses = self.model.generate(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                num_return_sequences=1
            )
        
        if isinstance(prompt, str):
            return responses[0] if responses else ""
        return responses
    
    def update_strength(self, new_strength: float):
        """
        Update intervention strength and recompute vectors.
        
        Useful for hyperparameter search without reinitializing.
        
        Args:
            new_strength: New intervention strength value.
        """
        self.config.intervention_strength = new_strength
        self._intervention_vectors = self._precompute_interventions()
    
    def get_intervention_stats(self) -> Dict:
        """
        Get statistics about the intervention vectors.
        
        Returns:
            Dictionary with mean, std, max values per layer.
        """
        stats = {}
        for name, vec in self._intervention_vectors.items():
            stats[name] = {
                "mean": vec.mean().item(),
                "std": vec.std().item(),
                "max": vec.abs().max().item(),
                "l2_norm": vec.norm(p=2).item(),
            }
        return stats
