"""
Unified model wrapper that supports multiple LLM architectures via HuggingFace transformers.

This module provides a clean abstraction layer for:
- Loading models from HuggingFace Hub
- Extracting activations via forward hooks
- Applying interventions during inference

Directly patches HuggingFace models without custom implementations.
"""
import torch
from torch import nn
from typing import Dict, List, Optional, Any, Callable, Union
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from contextlib import contextmanager
from config import ModelType, ExperimentConfig


class ModelWrapper:
    """
    A unified wrapper for LLM models that provides:
    - Consistent activation extraction interface across architectures
    - Support for Llama3, Qwen3, GLM, Gemma3
    - Hook-based intervention capabilities
    """
    
    # Architecture-specific layer name patterns
    # Format: "layers_template": how to access layer i, component c
    LAYER_PATTERNS = {
        ModelType.LLAMA3: {
            "layers_template": "model.layers.{layer_idx}",
            "attn": "self_attn",
            "mlp": "mlp",
            "attn_out": "self_attn.o_proj",
            "mlp_act": "mlp.act_fn",
            "mlp_gate": "mlp.gate_proj",
            "mlp_up": "mlp.up_proj",
            "mlp_down": "mlp.down_proj",
            "hidden": "mlp",
        },
        ModelType.QWEN3: {
            "layers_template": "model.layers.{layer_idx}",
            "attn": "self_attn",
            "mlp": "mlp",
            "attn_out": "self_attn.o_proj",
            "mlp_act": "mlp.act_fn",
            "mlp_gate": "mlp.gate_proj",
            "mlp_up": "mlp.up_proj",
            "mlp_down": "mlp.down_proj",
            "hidden": "mlp",
        },
        ModelType.GLM: {
            "layers_template": "transformer.encoder.layers.{layer_idx}",
            "attn": "self_attention",
            "mlp": "mlp",
            "attn_out": "self_attention.dense",
            "mlp_act": "mlp.activation_func",
            "mlp_gate": "mlp.dense_h_to_4h",
            "mlp_up": "mlp.dense_h_to_4h",
            "mlp_down": "mlp.dense_4h_to_h",
            "hidden": "mlp",
        },
        ModelType.GEMMA3: {
            "layers_template": "model.layers.{layer_idx}",
            "attn": "self_attn",
            "mlp": "mlp",
            "attn_out": "self_attn.o_proj",
            "mlp_act": "mlp.act_fn",
            "mlp_gate": "mlp.gate_proj",
            "mlp_up": "mlp.up_proj",
            "mlp_down": "mlp.down_proj",
            "hidden": "mlp",
        },
    }
    
    def __init__(self, config: ExperimentConfig):
        """
        Initialize model wrapper.
        
        Args:
            config: Experiment configuration containing model settings.
        """
        self.config = config
        self.device = torch.device(config.device)
        
        # Determine torch dtype
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        self.dtype = dtype_map.get(config.dtype, torch.float16)
        
        # Load tokenizer
        print(f"Loading tokenizer: {config.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_name,
            trust_remote_code=True,
            padding_side="left"
        )
        
        # Set padding token if not exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # Load model
        print(f"Loading model: {config.model_name}")
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            dtype=self.dtype,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()

        # Disable gradient computation globally
        for param in self.model.parameters():
            param.requires_grad = False

        # Get actual device (may be different from config.device when using device_map)
        # Find the device of the first parameter
        first_param = next(self.model.parameters())
        self.device = first_param.device
        
        # Detect model architecture
        self._detect_architecture()
        
        # Storage for hook activations
        self._activations: Dict[str, torch.Tensor] = {}
        
        print(f"Model loaded: {self.model_type.value}, {self.num_layers} layers")
    
    def _detect_architecture(self):
        """Auto-detect model architecture and number of layers."""
        # Use config model_type if available
        self.model_type = self.config.model_type
        
        # Auto-detect from model name if needed
        if self.model_type is None:
            name_lower = self.config.model_name.lower()
            if "llama" in name_lower:
                self.model_type = ModelType.LLAMA3
            elif "qwen" in name_lower:
                self.model_type = ModelType.QWEN3
            elif "glm" in name_lower or "chatglm" in name_lower:
                self.model_type = ModelType.GLM
            elif "gemma" in name_lower:
                self.model_type = ModelType.GEMMA3
            else:
                # Default to Llama-style architecture
                self.model_type = ModelType.LLAMA3
        
        # Get number of layers
        model_config = self.model.config
        self.num_layers = getattr(
            model_config,
            "num_hidden_layers",
            getattr(model_config, "num_layers", 32)
        )
        
        # Get hidden size
        self.hidden_size = getattr(
            model_config,
            "hidden_size",
            getattr(model_config, "d_model", 4096)
        )
        
        # Set layer pattern
        self.layer_pattern = self.LAYER_PATTERNS[self.model_type]
    
    def get_layer_name(self, layer_idx: int, component: str) -> str:
        """
        Get the full module path for a specific layer component.
        
        Args:
            layer_idx: Layer index (0-indexed).
            component: Component type ('attn', 'mlp', 'attn_out', 'mlp_act', 'hidden').
        
        Returns:
            Full module path string (e.g., 'model.layers.0.mlp.act_fn').
        """
        base = self.layer_pattern["layers_template"].format(layer_idx=layer_idx)
        component_path = self.layer_pattern.get(component, component)
        return f"{base}.{component_path}"
    
    def _get_module(self, name: str) -> nn.Module:
        """
        Get a module by its dot-separated path.
        
        Args:
            name: Module path (e.g., 'model.layers.0.mlp').
        
        Returns:
            The requested nn.Module.
        """
        module = self.model
        for part in name.split('.'):
            if part.isdigit():
                module = module[int(part)]
            else:
                module = getattr(module, part)
        return module
    
    def _make_hook(self, name: str) -> Callable:
        """
        Create a forward hook function that stores activations.
        
        Args:
            name: Key to store activation under.
        
        Returns:
            Hook function.
        """
        def hook(module: nn.Module, input: Any, output: Any):
            # Handle tuple outputs (some layers return tuples)
            out = output[0] if isinstance(output, tuple) else output
            self._activations[name] = out.detach()
        return hook
    
    @contextmanager
    def register_hooks(
        self,
        layer_indices: List[int],
        components: List[str] = None
    ):
        """
        Context manager to register forward hooks for activation extraction.
        
        Args:
            layer_indices: List of layer indices to hook.
            components: List of components to extract ('mlp_act', 'attn_out', etc.).
        
        Yields:
            Dictionary mapping layer names to activations after forward pass.
        
        Example:
            with model.register_hooks([0, 1, 2], ['mlp_act']) as acts:
                model.model(inputs)
            # acts now contains activations
        """
        if components is None:
            components = ["mlp_act"]
        
        self._activations.clear()
        handles = []
        
        try:
            # Register hooks for each layer and component
            for layer_idx in layer_indices:
                for component in components:
                    name = self.get_layer_name(layer_idx, component)
                    try:
                        module = self._get_module(name)
                        handle = module.register_forward_hook(self._make_hook(name))
                        handles.append(handle)
                    except (AttributeError, KeyError) as e:
                        print(f"Warning: Could not hook {name}: {e}")
            
            yield self._activations
            
        finally:
            # Clean up all hooks
            for handle in handles:
                handle.remove()
            self._activations.clear()
    
    @torch.no_grad()
    def generate(
        self,
        prompt: Union[str, List[str]],
        max_new_tokens: int = 32,
        temperature: float = 1.0,
        top_p: float = 0.9,
        do_sample: bool = True,
        num_return_sequences: int = 1
    ) -> List[str]:
        """
        Generate responses for a given prompt or batch of prompts.
        
        Args:
            prompt: Input prompt string or list of prompt strings.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature (higher = more diverse).
            top_p: Nucleus sampling probability threshold.
            do_sample: Whether to use sampling (False = greedy decoding).
            num_return_sequences: Number of sequences to return.
        
        Returns:
            List of generated response strings (prompt excluded).
        """
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        ).to(self.device)
        
        # Generation config
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "num_return_sequences": num_return_sequences,
        }
        
        if do_sample:
            gen_kwargs.update({
                "do_sample": True,
                "temperature": temperature,
                "top_p": top_p,
            })
        else:
            gen_kwargs["do_sample"] = False
        
        outputs = self.model.generate(**inputs, **gen_kwargs)
        
        # Decode and extract only the new tokens (exclude prompt)
        # Using input_ids.shape[1] is correct because padding is on the left
        prompt_len = inputs["input_ids"].shape[1]
        responses = []
        for output in outputs:
            response = self.tokenizer.decode(
                output[prompt_len:],
                skip_special_tokens=True
            )
            responses.append(response.strip())
        
        return responses
    
    @torch.no_grad()
    def get_activations(
        self,
        text: Union[str, List[str]],
        layer_indices: List[int],
        components: List[str] = None,
        token_position: int = -1
    ) -> Dict[str, torch.Tensor]:
        """
        Extract activations for specific layers and components.
        
        Supports both single text and batch of texts.
        
        Args:
            text: Input text string or list of text strings.
            layer_indices: Layers to extract from.
            components: Components to extract ('mlp_act', 'attn_out', etc.).
            token_position: Token position to extract (-1 for last token).
                Note: With left padding, -1 always gets the last real token.
        
        Returns:
            Dictionary mapping layer names to activation tensors.
            - Single input: [hidden_dim]
            - Batch input: [batch_size, hidden_dim]
        """
        if components is None:
            components = ["mlp_act"]
        
        # Check if input is a single string or batch
        is_single = isinstance(text, str)
        
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,  # Enable padding for batch support
            truncation=True,
            max_length=2048
        ).to(self.device)
        
        with self.register_hooks(layer_indices, components) as activations:
            self.model(**inputs)
            
            # Extract activations at specified token position
            # Must extract INSIDE with block before finally clears _activations
            # Note: With padding_side="left", token_position=-1 is always valid
            result = {}
            for name, act in activations.items():
                # act shape: [batch, seq_len, hidden_dim]
                if is_single:
                    # Single input: return [hidden_dim]
                    result[name] = act[0, token_position, :].cpu().float()
                else:
                    # Batch input: return [batch_size, hidden_dim]
                    result[name] = act[:, token_position, :].cpu().float()
        
        return result
    
    @torch.no_grad()
    def forward(self, text: str) -> torch.Tensor:
        """
        Run forward pass and return logits.
        
        Args:
            text: Input text.
        
        Returns:
            Logits tensor [seq_len, vocab_size].
        """
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(self.device)
        
        outputs = self.model(**inputs)
        return outputs.logits[0]
