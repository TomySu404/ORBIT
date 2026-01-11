"""
Model wrappers and activation hooks for various LLM architectures.

Supports: Llama3, Qwen3, GLM, Gemma3 via HuggingFace transformers.
"""

from .wrapper import ModelWrapper

__all__ = ["ModelWrapper"]
