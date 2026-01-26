"""
Main entry point for ORBIT (On-distribution Rollout-based Behavioral Intervention Technique).
Implements two key improvements:
1. Rollout-based contrastive pair generation (ORC)
2. Continuous soft scaling (CSS)

Usage:
    # Single GPU
    python main.py --model meta-llama/Llama-3.1-8B-Instruct --dataset copa
    
    # Multi-GPU with torchrun (recommended)
    torchrun --nproc_per_node=8 main.py --parallel_gpus --model <model> --dataset <dataset>
    
    # Multi-GPU with launch script
    bash run_distributed.sh --model <model> --dataset <dataset>
    
    # Multi-GPU with torch.distributed.launch
    python -m torch.distributed.launch --nproc_per_node=8 main.py --parallel_gpus --model <model> --dataset <dataset>
"""
import json
import torch
import torch.distributed as dist
import argparse
import random
import copy
import numpy as np
import itertools
import os
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from tqdm import tqdm
from typing import List, Tuple, Dict, Optional, Any
from config import (
    ExperimentConfig,
    RolloutConfig,
    InterventionConfig,
    LayerScope
)
from models.wrapper import ModelWrapper
from steering.rollout import RolloutGenerator
from steering.diff_vector import ContinuousDiffCalculator, DiffVectorResult
from steering.intervention import ActivationIntervention
from data.loader import DatasetLoader
from utils.metrics import Evaluator, EvaluationResult

def serialize_steering_vectors(steering_data: Dict) -> Dict:
    """
    Convert steering vectors to JSON-serializable format.

    Args:
        steering_data: Dictionary containing diff_result, stats, and layer_indices

    Returns:
        Dictionary with serialized tensors and metadata
    """
    diff_result = steering_data["diff_result"]

    serialized = {
        "stats": steering_data["stats"],
        "layer_indices": steering_data["layer_indices"],
        "diff_vectors": {},
        "scaling_weights": {},
        "tensor_shapes": {},
        "tensor_dtypes": {}
    }

    # Serialize diff_vectors and scaling_weights
    for layer_key in diff_result.diff_vectors.keys():
        diff_vec = diff_result.diff_vectors[layer_key]
        scale_weight = diff_result.scaling_weights[layer_key]

        # Convert to CPU numpy arrays, then to lists for JSON serialization
        serialized["diff_vectors"][layer_key] = diff_vec.cpu().numpy().tolist()
        serialized["scaling_weights"][layer_key] = scale_weight.cpu().numpy().tolist()

        # Store metadata for reconstruction
        serialized["tensor_shapes"][layer_key] = list(diff_vec.shape)
        serialized["tensor_dtypes"][layer_key] = str(diff_vec.dtype)

    # Add aggregation statistics
    serialized["aggregation_stats"] = {
        "sample_count": diff_result.sample_count,
        "pair_count": diff_result.pair_count,
        "reread_sample_count": diff_result.reread_sample_count
    }

    return serialized

# Distributed training utilities
def init_distributed(rank: int, world_size: int, backend: str = "nccl"):
    """Initialize distributed process group."""
    # Don't override MASTER_ADDR and MASTER_PORT if already set by torchrun
    if 'MASTER_ADDR' not in os.environ:
        os.environ['MASTER_ADDR'] = 'localhost'
    if 'MASTER_PORT' not in os.environ:
        os.environ['MASTER_PORT'] = '12355'
    
    # NOTE: all_gather_object/gather_object may take a long time when results are large
    # or when one rank experiences long-tail latency. Increase timeout to reduce
    # spurious NCCL watchdog kills.
    dist.init_process_group(
        backend=backend,
        rank=rank,
        world_size=world_size,
        timeout=timedelta(hours=2)
    )
    torch.cuda.set_device(rank)

def cleanup_distributed():
    """Cleanup distributed process group."""
    if dist.is_initialized():
        dist.destroy_process_group()

def get_rank() -> int:
    """Get current process rank."""
    if dist.is_initialized():
        return dist.get_rank()
    return 0

def get_world_size() -> int:
    """Get world size (number of processes)."""
    if dist.is_initialized():
        return dist.get_world_size()
    return 1

def is_main_process() -> bool:
    """Check if current process is main process (rank 0)."""
    return get_rank() == 0

def print_rank0(*args, **kwargs):
    """Print only on rank 0 to avoid duplicate output."""
    if is_main_process():
        print(*args, **kwargs)

def tqdm_rank0(*args, **kwargs):
    """Create tqdm progress bar only on rank 0."""
    if is_main_process():
        return tqdm(*args, **kwargs)
    else:
        # Return a dummy iterator that doesn't show progress
        class DummyTqdm:
            def __init__(self, iterable=None, *args, **kwargs):
                if iterable is not None:
                    self.iterable = iterable
                elif 'total' in kwargs:
                    self.iterable = range(kwargs['total'])
                else:
                    self.iterable = []
            def __iter__(self):
                return iter(self.iterable)
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
            def update(self, n=1):
                pass
            def close(self):
                pass
        # Handle both positional and keyword arguments
        if args:
            return DummyTqdm(iterable=args[0], **kwargs)
        else:
            return DummyTqdm(**kwargs)

def _diff_result_to_cpu(diff_result: DiffVectorResult) -> DiffVectorResult:
    """Move DiffVectorResult tensors to CPU for safe serialization/communication."""
    cpu_diff_vectors = {k: v.detach().cpu() for k, v in diff_result.diff_vectors.items()}
    cpu_scaling_weights = {k: v.detach().cpu() for k, v in diff_result.scaling_weights.items()}
    return DiffVectorResult(
        diff_vectors=cpu_diff_vectors,
        scaling_weights=cpu_scaling_weights,
        sample_count=diff_result.sample_count,
        pair_count=diff_result.pair_count,
        reread_sample_count=diff_result.reread_sample_count
    )

def _gather_results_all_ranks(local_obj: Any, world_size: int) -> List[Any]:
    """
    Gather Python objects from all ranks using all_gather_object.
    
    This is more reliable than gather_object which may have issues in some PyTorch versions.
    All ranks will have the complete result list after this call.
    """
    if world_size <= 1 or not dist.is_initialized():
        return [local_obj]

    result_list = [None] * world_size
    dist.all_gather_object(result_list, local_obj)
    return result_list

def evaluation_result_to_dict(result: EvaluationResult) -> Dict:
    """Convert EvaluationResult to dictionary for JSON serialization."""
    return {
        "accuracy": result.accuracy,
        "total": result.total,
        "correct": result.correct,
        "predictions": result.predictions
    }

def _merge_steering_results(chunk_results: List[Dict], intervention_config: InterventionConfig) -> Dict:
    """Merge steering vector results from multiple GPU processes."""
    if not chunk_results:
        raise RuntimeError("No steering results to merge")

    if len(chunk_results) == 1:
        return chunk_results[0]

    # Aggregate statistics
    merged_stats = {
        "total_samples": sum(r["stats"]["total_samples"] for r in chunk_results),
        "samples_with_rollout_correct": sum(r["stats"]["samples_with_rollout_correct"] for r in chunk_results),
        "samples_with_reread": sum(r["stats"]["samples_with_reread"] for r in chunk_results),
        "total_pairs": sum(r["stats"]["total_pairs"] for r in chunk_results),
        "reread_pairs": sum(r["stats"]["reread_pairs"] for r in chunk_results),
        "skipped_samples": sum(r["stats"]["skipped_samples"] for r in chunk_results)
    }

    # Properly merge diff_results from all chunks using weighted average
    # Weight by pair_count to ensure correct aggregation
    total_pair_count = sum(r["diff_result"].pair_count for r in chunk_results)
    if total_pair_count == 0:
        raise RuntimeError("No pairs found in any chunk results")

    # Get layer names from first chunk (all should have same layers)
    first_diff_result = chunk_results[0]["diff_result"]
    layer_names = list(first_diff_result.diff_vectors.keys())

    # Merge diff_vectors using weighted average based on pair_count
    merged_diff_vectors = {}

    # Pre-compute all weights to avoid repeated division
    chunk_weights = []
    valid_chunks = []
    for chunk_result in chunk_results:
        diff_result = chunk_result["diff_result"]
        weight = diff_result.pair_count / total_pair_count if total_pair_count > 0 else 0.0
        if weight > 0 and diff_result.pair_count > 0:
            chunk_weights.append(weight)
            valid_chunks.append(chunk_result)

    # Vectorized merge for better performance
    for layer_name in layer_names:
        # Collect tensors for this layer from valid chunks
        layer_tensors = [chunk["diff_result"].diff_vectors[layer_name] for chunk in valid_chunks]

        if layer_tensors:
            # Stack tensors: [num_valid_chunks, hidden_dim]
            stacked = torch.stack(layer_tensors)
            # Weight tensor: [num_valid_chunks]
            weights_tensor = torch.tensor(chunk_weights)

            # Vectorized weighted sum: sum over chunks
            weighted_sum = torch.sum(weights_tensor.unsqueeze(1) * stacked, dim=0)

            merged_diff_vectors[layer_name] = weighted_sum
        else:
            # Fallback
            first_tensor = chunk_results[0]["diff_result"].diff_vectors[layer_name]
            merged_diff_vectors[layer_name] = torch.zeros_like(first_tensor)

    # Recompute scaling_weights based on merged diff_vectors
    # Don't merge scaling_weights directly - they must be computed from the merged diffs
    eps = 1e-8  # Numerical stability
    scaling_method = intervention_config.scaling_method
    merged_scaling_weights = {}

    # OPTIMIZED: Batch GPU transfer and scaling computation
    # Instead of transferring each layer individually (slow due to sync overhead),
    # stack all layers, transfer once, compute all scalings, transfer back once
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Stack all diff vectors: [num_layers, hidden_dim] - single GPU transfer
    all_diffs_list = [merged_diff_vectors[layer_name] for layer_name in layer_names]
    all_diffs_stacked = torch.stack(all_diffs_list).to(device)

    if scaling_method == "max_norm":
        # Compute max for each layer: [num_layers, 1]
        max_vals = torch.max(torch.abs(all_diffs_stacked), dim=1, keepdim=True).values
        max_vals = torch.clamp(max_vals, min=eps)  # Avoid division by zero
        all_scaled = all_diffs_stacked / max_vals

    elif scaling_method == "l2_norm":
        # Compute L2 norm for each layer: [num_layers, 1]
        norms = torch.norm(all_diffs_stacked, p=2, dim=1, keepdim=True)
        norms = torch.clamp(norms, min=eps)
        all_scaled = all_diffs_stacked / norms

    elif scaling_method == "softmax":
        # Softmax per layer
        abs_diffs = torch.abs(all_diffs_stacked)
        weights = torch.softmax(abs_diffs, dim=1)
        hidden_dim = all_diffs_stacked.shape[1]
        all_scaled = weights * torch.sign(all_diffs_stacked) * hidden_dim

    else:
        raise ValueError(f"Unknown scaling method: {scaling_method}")

    # Single transfer back to CPU and unpack results
    all_scaled_cpu = all_scaled.cpu()
    for i, layer_name in enumerate(layer_names):
        merged_scaling_weights[layer_name] = all_scaled_cpu[i]
    
    # Aggregate sample and pair counts
    merged_sample_count = sum(r["diff_result"].sample_count for r in chunk_results)
    merged_pair_count = sum(r["diff_result"].pair_count for r in chunk_results)
    merged_reread_count = sum(r["diff_result"].reread_sample_count for r in chunk_results)
    
    # Create merged DiffVectorResult
    merged_diff_result = DiffVectorResult(
        diff_vectors=merged_diff_vectors,
        scaling_weights=merged_scaling_weights,
        sample_count=merged_sample_count,
        pair_count=merged_pair_count,
        reread_sample_count=merged_reread_count
    )
    
    merged_result = {
        "diff_result": merged_diff_result,
        "stats": merged_stats,
        "layer_indices": chunk_results[0]["layer_indices"]
    }

    print(f"📊 Merged steering results from {len(chunk_results)} chunks")
    print(f"   Total samples: {merged_stats['total_samples']}")
    print(f"   Total pairs: {merged_stats['total_pairs']}")

    return merged_result

def _combine_evaluation_results(results: List[EvaluationResult]) -> EvaluationResult:
    """Combine evaluation results from multiple GPU processes."""
    if not results:
        raise RuntimeError("No evaluation results to combine")

    if len(results) == 1:
        return results[0]

    # Combine all predictions and calculate overall accuracy
    all_predictions = []
    total_correct = 0
    total_samples = 0

    for result in results:
        all_predictions.extend(result.predictions)
        total_correct += result.correct
        total_samples += result.total

    overall_accuracy = total_correct / total_samples if total_samples > 0 else 0.0

    # Create combined result
    from utils.metrics import EvaluationResult as EvalResult
    combined_result = EvalResult(
        accuracy=overall_accuracy,
        total=total_samples,
        correct=total_correct,
        predictions=all_predictions
    )

    return combined_result
def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
def tune_hyperparameters(
    model, train_data, dev_data,
    rollout_config, intervention_config,
    args, seed: int, num_gpus: int = 1
):
    """
    Tune hyperparameters on dev set to avoid overfitting to test set.
    Returns:
        tuple: (best_config, dev_results, steering_data, max_train_samples)
            - best_config: Best hyperparameter configuration found
            - dev_results: List of results for each config tried
            - steering_data: Pre-computed steering vectors (can be reused)
            - max_train_samples: Number of samples used for steering vectors
    """
    # Define hyperparameter search space
    param_grid = {
        'strength': [1e-3,0.01, 0.05, 0.08, 0.2, 0.5, 1, 2],
    }
    # Generate all combinations dynamically
    keys, values = zip(*param_grid.items())
    all_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    # Limit combinations for efficiency
    max_combinations = 12
    # Randomly sample combinations if too many
    if len(all_combinations) > max_combinations:
        random.seed(seed + 2000)  # Different seed for hyperparameter sampling
        all_combinations = random.sample(all_combinations, max_combinations)
    print(f"Testing {len(all_combinations)} hyperparameter combinations on dev set...")
    
    # OPTIMIZATION: Pre-compute steering vectors once (doesn't depend on strength)
    # The diff_result (diff_vectors and scaling_weights) only depends on rollout
    # and diff computation, not on intervention_strength
    # In distributed mode, each process already has the model loaded, so we can use multi-GPU
    print_rank0("\n📊 Pre-computing steering vectors (shared across all configs)...")
    max_train_samples = min(args.max_train or 100, args.max_tune_samples)
    steering_data = build_steering_vectors(
        model, train_data,
        rollout_config, intervention_config,  # Use base config for diff_vector computation
        reread_weight=args.reread_weight,
        max_samples=max_train_samples,
        show_progress=True,
        diff_batch_size=args.diff_batch_size,
        format_type=args.format_type,
        num_gpus=num_gpus  # Use multi-GPU in distributed mode
    )
    
    best_score = 0.0
    best_config = None
    dev_results = []
    for i, config in enumerate(all_combinations):
        # Get current parameters, falling back to initial config if not in grid
        curr_strength = config.get(
            'strength', intervention_config.intervention_strength)
        curr_layers = config.get('num_layers', intervention_config.num_layers)
        curr_components = config.get('components', intervention_config.components)
        print_rank0(f"\n🔧 Config {i+1}/{len(all_combinations)}: "
              f"strength={curr_strength}, layers={curr_layers}, "
              f"components={curr_components}")
        try:
            # Create config with current hyperparameters
            current_intervention_config = InterventionConfig(
                layer_scope=intervention_config.layer_scope,
                num_layers=curr_layers,
                scaling_method=intervention_config.scaling_method,
                intervention_strength=curr_strength,  # Only strength changes
                components=curr_components,
                prefill_only=intervention_config.prefill_only
            )
            # Reuse pre-computed steering vectors, only create intervention with new strength
            intervention = ActivationIntervention(
                model, current_intervention_config, steering_data["diff_result"]
            )
            # Evaluate on dev set
            # In distributed mode, each process already has the model loaded, so we can use multi-GPU
            dev_result = evaluate_model(
                model, dev_data,
                intervention=intervention,
                max_samples=len(dev_data),  # Use all dev samples
                max_new_tokens=args.max_new_tokens,
                batch_size=args.batch_size,
                show_progress=False,
                desc=f"Dev {i+1}",
                verbose=args.verbose,
                num_gpus=num_gpus  # Use multi-GPU in distributed mode
            )
            score = dev_result.accuracy
            dev_results.append({
                'config': config,
                'accuracy': score,
                'correct': dev_result.correct,
                'total': dev_result.total
            })
            print_rank0(f"Dev accuracy: {score:.4f}")
            # Update best
            if score >= best_score:
                best_score = score
                best_config = {
                    'strength': curr_strength,
                    'num_layers': curr_layers,
                    'components': curr_components
                }
        except Exception as e:
            print_rank0(f"❌ Failed: {e}")
            dev_results.append({
                'config': config,
                'accuracy': 0.0,
                'correct': 0,
                'total': len(dev_data),
                'error': str(e)
            })
    print_rank0(f"\n✅ Tuning completed. Best score: {best_score:.4f}")
    return best_config, dev_results, steering_data, max_train_samples


def build_steering_vectors(
    model: ModelWrapper,
    train_data: List[Tuple[str, str, str]],
    rollout_config: RolloutConfig,
    intervention_config: InterventionConfig,
    reread_weight: float = 0.5,
    max_samples: int = 150,
    show_progress: bool = True,
    diff_batch_size: int = 16,
    format_type: str = "generation",
    num_gpus: int = 1
) -> Dict:
    """
    Build steering vectors from training data using rollout method.
    This implements the core of our improved approach:
    1. Generate multiple responses per sample via rollout
    2. Classify into correct/incorrect
    3. Form contrastive pairs
    4. Compute continuous difference vectors
    5. Aggregate with optional re-read sample down-weighting
    Args:
        model: Model wrapper instance.
        train_data: List of (question, correct_answer, wrong_answer) tuples.
        rollout_config: Configuration for rollout generation.
        intervention_config: Configuration for intervention.
        reread_weight: Weight for re-read samples (< 1 to down-weight).
        max_samples: Maximum number of training samples.
        show_progress: Whether to show progress bar.
        num_gpus: Number of GPUs to use for parallel processing.
    Returns:
        Dictionary containing:
        - diff_result: DiffVectorResult with aggregated vectors
        - stats: Statistics about the build process
        - layer_indices: List of layer indices being intervened
    """
    world_size = get_world_size()
    rank = get_rank()
    
    # Use distributed if available, otherwise fall back to single GPU
    use_distributed = world_size > 1 and num_gpus > 1
    
    if not use_distributed:
        return _build_steering_vectors_single_gpu(
            model, train_data, rollout_config, intervention_config,
            reread_weight, max_samples, show_progress, diff_batch_size, format_type
        )

    # Distributed processing: each process handles its own chunk
    print_rank0(f"🔥 Using {world_size} GPUs for parallel rollout generation")
    
    # Split data across processes
    samples = train_data[:max_samples]
    print_rank0(f"Total training samples: {len(samples)}")
    
    if len(samples) < world_size:
        print_rank0(f"Too few samples ({len(samples)}) for {world_size} GPUs, falling back to single GPU")
        return _build_steering_vectors_single_gpu(
            model, train_data, rollout_config, intervention_config,
            reread_weight, max_samples, show_progress, diff_batch_size, format_type
        )

    # Distribute data across ranks
    chunk_size = len(samples) // world_size
    start_idx = rank * chunk_size
    end_idx = (rank + 1) * chunk_size if rank < world_size - 1 else len(samples)
    local_chunk = samples[start_idx:end_idx]
    
    # Print from every rank to make load imbalance/debugging visible
    print(f"[rank{rank}] {len(local_chunk)} samples (indices {start_idx}:{end_idx})")
    
    # Process local chunk (model is already loaded on this process)
    local_result = None
    try:
        local_result = _build_steering_vectors_single_gpu(
            model, local_chunk, rollout_config, intervention_config,
            reread_weight, None, show_progress and is_main_process(), diff_batch_size, format_type
        )

        # Move tensors to CPU before distributed communication (safer + less GPU pressure)
        local_result_for_gather = dict(local_result)
        local_result_for_gather["diff_result"] = _diff_result_to_cpu(local_result["diff_result"])

        payload = {"ok": True, "rank": rank, "result": local_result_for_gather}
    except Exception as e:
        payload = {
            "ok": False,
            "rank": rank,
            "error": repr(e),
            "traceback": traceback.format_exc()
        }

    # Gather results from all ranks using all_gather_object (more reliable)
    gathered = _gather_results_all_ranks(payload, world_size)

    # Check for errors across all ranks (all ranks have the complete list now)
    has_error = any((p is None) or (not p.get("ok", False)) for p in gathered)
    if has_error:
        if is_main_process():
            for p in gathered:
                if p is None or p.get("ok", False):
                    continue
                print_rank0(f"[rank{p.get('rank')}] build_steering_vectors failed: {p.get('error')}")
                print_rank0(p.get("traceback", ""))
        raise RuntimeError("Distributed build_steering_vectors failed on at least one rank. See rank0 logs.")

    # Merge results on rank 0
    if is_main_process():
        result_list = [p["result"] for p in gathered]
        merged_result = _merge_steering_results(result_list, intervention_config)
        return merged_result

    # Return local result for non-main processes (though it won't be used)
    return local_result


def _build_steering_vectors_single_gpu(
    model_or_config,
    train_data: List[Tuple[str, str, str]],
    rollout_config: RolloutConfig,
    intervention_config: InterventionConfig,
    reread_weight: float = 0.5,
    max_samples: int = 150,
    show_progress: bool = True,
    diff_batch_size: int = 16,
    format_type: str = "generation"
) -> Dict:
    """Single GPU version of build_steering_vectors."""
    # Handle both ModelWrapper and ExperimentConfig inputs
    # (for backward compatibility with single-GPU calls)
    if isinstance(model_or_config, ModelWrapper):
        model = model_or_config
    else:
        # Recreate model in this process (cannot pickle model objects)
        model = ModelWrapper(model_or_config)

    rollout_gen = RolloutGenerator(model, rollout_config)
    diff_calc = ContinuousDiffCalculator(
        model, intervention_config, format_type=format_type, enable_thinking=rollout_config.enable_thinking)

    # Collect all contrastive pairs first
    all_pairs_data = []  # List of (question, positive, negative, used_reread)
    pairs_per_question = []  # Track number of pairs per question for grouped normalization
    stats = {
        "total_samples": 0,
        "samples_with_rollout_correct": 0,
        "samples_with_reread": 0,
        "total_pairs": 0,
        "reread_pairs": 0,
        "skipped_samples": 0
    }
    # train_data is already chunked in multi-GPU mode, so use all of it
    samples = train_data[:max_samples] if max_samples else train_data
    iterator = tqdm_rank0(samples, desc="Generating rollouts") if show_progress else samples

    # Phase 1: Generate rollouts and collect contrastive pairs
    for question, correct_ans, wrong_ans in iterator:
        stats["total_samples"] += 1
        # Generate rollouts and build contrastive pairs
        result = rollout_gen.build_contrastive_pairs(
            question, correct_ans, wrong_ans
        )
        # Skip if no pairs could be formed
        if not result.contrastive_pairs:
            stats["skipped_samples"] += 1
            continue
        # Track statistics
        if result.used_reread_mechanism:
            stats["samples_with_reread"] += 1
        else:
            stats["samples_with_rollout_correct"] += 1
        # Record how many pairs this question contributes (for grouped normalization)
        pairs_per_question.append(len(result.contrastive_pairs))
        # Collect pairs for batch processing
        for pair in result.contrastive_pairs:
            all_pairs_data.append((
                question,
                pair.positive,
                pair.negative,
                pair.used_reread
            ))
            stats["total_pairs"] += 1
            if pair.used_reread:
                stats["reread_pairs"] += 1
    # Phase 2: Batch compute diffs
    all_diffs = []
    reread_flags = []
    if all_pairs_data:
        # Process in batches
        n_pairs = len(all_pairs_data)
        n_batches = (n_pairs + diff_batch_size - 1) // diff_batch_size
        batch_iter = range(n_batches)
        if show_progress:
            batch_iter = tqdm_rank0(batch_iter, desc="Computing diffs (batch)")
        for batch_idx in batch_iter:
            start_idx = batch_idx * diff_batch_size
            end_idx = min(start_idx + diff_batch_size, n_pairs)
            batch_data = all_pairs_data[start_idx:end_idx]
            # Unpack batch data
            questions = [d[0] for d in batch_data]
            positives = [d[1] for d in batch_data]
            negatives = [d[2] for d in batch_data]
            batch_reread = [d[3] for d in batch_data]
            try:
                # Batch compute diffs
                batch_diffs = diff_calc.compute_batch_pair_diffs(
                    questions, positives, negatives,
                    components=intervention_config.components
                )
                all_diffs.extend(batch_diffs)
                reread_flags.extend(batch_reread)
            except Exception as e:
                if show_progress:
                    print_rank0(f"Warning: Batch diff failed: {e}")
                # Fallback to single pair processing
                for q, pos, neg, rr in batch_data:
                    try:
                        diff = diff_calc.compute_pair_diff(
                            q, pos, neg,
                            components=intervention_config.components
                        )
                        all_diffs.append(diff)
                        reread_flags.append(rr)
                    except Exception as e2:
                        if show_progress:
                            print_rank0(f"Warning: Single diff failed: {e2}")
    if not all_diffs:
        raise RuntimeError(
            "No valid difference vectors computed. Check your data and model.")
    # Aggregate all diffs with re-read weighting
    # Choose aggregation method based on configuration
    if intervention_config.use_grouped_normalization:
        # Per-question normalization before global averaging ("One Question One Vote")
        diff_result = diff_calc.aggregate_diffs_grouped(
            all_diffs,
            pairs_per_question,
            reread_flags,
            reread_weight=reread_weight
        )
        print_rank0("   Using grouped normalization (per-question normalization)")
    else:
        # Standard global averaging then normalization
        diff_result = diff_calc.aggregate_diffs(
            all_diffs,
            reread_flags,
            reread_weight=reread_weight
        )
    print_rank0("\n📊 Steering Vector Statistics:")
    print_rank0(f"   Total samples processed: {stats['total_samples']}")
    print_rank0(f"   Samples with rollout correct: {stats['samples_with_rollout_correct']}")
    print_rank0(f"   Samples using re-read: {stats['samples_with_reread']}")
    print_rank0(f"   Skipped samples: {stats['skipped_samples']}")
    print_rank0(f"   Total contrastive pairs: {stats['total_pairs']}")
    print_rank0(f"   Re-read pairs: {stats['reread_pairs']}")
    print_rank0(f"   Layers intervened: {diff_calc.layer_indices}")
    return {
        "diff_result": diff_result,
        "stats": stats,
        "layer_indices": diff_calc.layer_indices
    }
def evaluate_model(
    model: ModelWrapper,
    test_data: List[Tuple[str, str, str]],
    intervention: Optional[ActivationIntervention] = None,
    max_samples: Optional[int] = None,
    max_new_tokens: int = 16,
    batch_size: int = 16,
    show_progress: bool = True,
    desc: str = "Evaluating",
    verbose: bool = False,
    num_gpus: int = 1
) -> EvaluationResult:
    """
    Evaluate model on test data with optional intervention using batch inference.
    Args:
        model: Model wrapper instance.
        test_data: List of (question, correct_answer, wrong_answer) tuples.
        intervention: Optional intervention handler.
        max_samples: Maximum test samples (None for all).
        max_new_tokens: Maximum number of new tokens to generate.
        batch_size: Batch size for inference.
        show_progress: Whether to show progress bar.
        desc: Description for progress bar.
        verbose: Whether to show per-sample results.
        num_gpus: Number of GPUs to use for parallel processing.
    Returns:
        Dictionary with evaluation results.
    """
    world_size = get_world_size()
    rank = get_rank()
    
    # Use distributed if available, otherwise fall back to single GPU
    use_distributed = world_size > 1 and num_gpus > 1
    
    if not use_distributed:
        return _evaluate_model_single_gpu(
            model, test_data, intervention, max_samples, max_new_tokens,
            batch_size, show_progress, desc, verbose
        )

    # Distributed processing: each process handles its own chunk
    print_rank0(f"🔥 Using {world_size} GPUs for parallel evaluation")
    
    # Split data across processes
    samples = test_data[:max_samples] if max_samples else test_data
    print_rank0(f"Total test samples: {len(samples)}")
    
    if len(samples) < world_size:
        print_rank0(f"Too few samples ({len(samples)}) for {world_size} GPUs, falling back to single GPU")
        return _evaluate_model_single_gpu(
            model, test_data, intervention, max_samples, max_new_tokens,
            batch_size, show_progress, desc, verbose
        )

    # Distribute data across ranks
    chunk_size = len(samples) // world_size
    start_idx = rank * chunk_size
    end_idx = (rank + 1) * chunk_size if rank < world_size - 1 else len(samples)
    local_chunk = samples[start_idx:end_idx]
    
    # Print from every rank to make load imbalance/debugging visible
    print(f"[rank{rank}] {len(local_chunk)} samples (indices {start_idx}:{end_idx})")
    
    # Process local chunk (model is already loaded on this process)
    local_result = None
    try:
        local_result = _evaluate_model_single_gpu(
            model, local_chunk, intervention, None, max_new_tokens,
            batch_size, show_progress and is_main_process(), f"{desc} Rank{rank}", verbose
        )
        payload = {"ok": True, "rank": rank, "result": local_result}
    except Exception as e:
        payload = {
            "ok": False,
            "rank": rank,
            "error": repr(e),
            "traceback": traceback.format_exc()
        }

    # Gather results from all ranks using all_gather_object (more reliable)
    gathered = _gather_results_all_ranks(payload, world_size)

    # Check for errors across all ranks (all ranks have the complete list now)
    has_error = any((p is None) or (not p.get("ok", False)) for p in gathered)
    if has_error:
        if is_main_process():
            for p in gathered:
                if p is None or p.get("ok", False):
                    continue
                print_rank0(f"[rank{p.get('rank')}] evaluate_model failed: {p.get('error')}")
                print_rank0(p.get("traceback", ""))
        raise RuntimeError("Distributed evaluate_model failed on at least one rank. See rank0 logs.")

    # Combine results from all ranks
    result_list = [p["result"] for p in gathered]
    combined_result = _combine_evaluation_results(result_list)
    return combined_result


def _evaluate_model_single_gpu(
    model_or_config,
    test_data: List[Tuple[str, str, str]],
    intervention_or_info=None,
    max_samples: Optional[int] = None,
    max_new_tokens: int = 16,
    batch_size: int = 16,
    show_progress: bool = True,
    desc: str = "Evaluating",
    verbose: bool = False
) -> EvaluationResult:
    """Single GPU version of evaluate_model."""
    # Handle both ModelWrapper and ExperimentConfig inputs
    # (for backward compatibility with single-GPU calls)
    if isinstance(model_or_config, ModelWrapper):
        model = model_or_config
        intervention = intervention_or_info
    else:
        # Recreate model in this process (cannot pickle model objects)
        model = ModelWrapper(model_or_config)
        
        # Recreate intervention if needed
        intervention = None
        if intervention_or_info:
            intervention = ActivationIntervention(
                model,
                intervention_or_info['config'],
                intervention_or_info['diff_result']
            )

    evaluator = Evaluator(verbose=verbose and is_main_process())
    # test_data is already chunked in multi-GPU mode, so use all of it
    samples = test_data[:max_samples] if max_samples else test_data
    # Process in batches
    num_batches = (len(samples) + batch_size - 1) // batch_size
    iterator = range(0, len(samples), batch_size)
    if show_progress:
        iterator = tqdm_rank0(iterator, total=num_batches, desc=desc)
    for i in iterator:
        batch = samples[i:i + batch_size]
        questions = [s[0] for s in batch]
        correct_answers = [s[1] for s in batch]
        try:
            if intervention:
                responses = intervention.generate_with_intervention(
                    questions,
                    max_new_tokens=max_new_tokens,
                    do_sample=False
                )
            else:
                responses = model.generate(
                    questions,
                    max_new_tokens=max_new_tokens,
                    do_sample=False
                )
            
            # Process results sample by sample to be robust
            for response, correct_ans in zip(responses, correct_answers):
                try:
                    evaluator.evaluate_single(response, correct_ans)
                except Exception as e:
                    print_rank0(f"Warning: Single sample evaluation failed: {e}")
        except Exception as e:
            if show_progress:
                print_rank0(f"Warning: Batch processing failed: {e}")
    result = evaluator.get_result()
    return result
def main():
    parser = argparse.ArgumentParser(
        description="ORBIT: On-distribution Rollout-based Behavioral Intervention Technique"
    )
    # Distributed training arguments
    parser.add_argument(
        "--local_rank", type=int, default=-1,
        help="Local rank for distributed training (set by torchrun/launch script)"
    )
    parser.add_argument(
        "--world_size", type=int, default=1,
        help="World size for distributed training (set by torchrun/launch script)"
    )
    # Model arguments
    parser.add_argument(
        "--model", type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="HuggingFace model name or path"
    )
    parser.add_argument(
        "--dtype", type=str,
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
        help="Model precision"
    )
    # Dataset arguments
    parser.add_argument(
        "--datasets", type=str, nargs="+",
        default=["sst2"],
        help="Dataset name (boolq, copa, sst2, sst5, mmlu, xnli, winogrande, gsm8k, math500, truthfulqa, ifeval, spider)"
    )
    parser.add_argument(
        "--data_root", type=str,
        default="./data",
        help="Root directory for datasets"
    )
    parser.add_argument(
        "--max_train", type=int,
        default=None,
        help="Maximum training samples"
    )
    parser.add_argument(
        "--max_test", type=int,
        default=None,
        help="Maximum test samples (None for all)"
    )
    parser.add_argument(
        "--max_new_tokens", type=int,
        default=4,
        help="Maximum new tokens for evaluation"
    )
    parser.add_argument(
        "--batch_size", type=int,
        default=16,
        help="Batch size for evaluation"
    )
    parser.add_argument(
        "--eval_baseline", action="store_true",
        default=True,
        help="Whether to run baseline evaluation"
    )
    parser.add_argument(
        "--no_baseline", action="store_false",
        dest="eval_baseline",
        help="Skip baseline evaluation"
    )
    # Rollout arguments
    parser.add_argument(
        "--num_rollouts", type=int,
        default=8,
        help="Number of rollout responses per sample"
    )
    parser.add_argument(
        "--temperature", type=float,
        default=0.8,
        help="Sampling temperature for rollouts"
    )
    parser.add_argument(
        "--top_p", type=float,
        default=0.9,
        help="Top-p sampling threshold"
    )
    parser.add_argument(
        "--no_reread", action="store_true",
        help="Disable re-read fallback mechanism"
    )
    parser.add_argument(
        "--reread_weight", type=float,
        default=0.5,
        help="Weight for re-read samples (< 1 to down-weight)"
    )
    parser.add_argument(
        "--diff_batch_size", type=int,
        default=32,
        help="Batch size for diff computation (pairs per batch)"
    )
    # Intervention arguments
    parser.add_argument(
        "--strength", type=float,
        default=0.2,
        help="Intervention strength"
    )
    parser.add_argument(
        "--scaling", type=str,
        default="max_norm",
        choices=["softmax", "l2_norm", "max_norm"],
        help="Scaling method for continuous weights"
    )
    parser.add_argument(
        "--layer_scope", type=str,
        default="first_n",
        choices=["all", "first_n", "last_n"],
        help="Layer scope for intervention"
    )
    parser.add_argument(
        "--num_layers", type=int,
        default=5,
        help="Number of layers for first_n/last_n scope"
    )
    parser.add_argument(
        "--components", type=str,
        default="mlp_act",
        help="Comma-separated list of components to intervene"
    )
    parser.add_argument(
        "--prefill_only", action="store_true",
        help="Only intervene during prefill phase, not during token generation"
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+",
        default=[42, 22, 52],
        help="Random seeds for multiple runs"
    )
    parser.add_argument(
        "--output_dir", type=str,
        default="./results",
        help="Directory to save results"
    )
    parser.add_argument(
        "--tune_hyperparams", action="store_true",
        help="Perform hyperparameter tuning on dev set before final evaluation"
    )
    parser.add_argument(
        "--dev_ratio", type=float,
        default=0.2,
        help="Ratio of test data to use for dev set (0.0-1.0)"
    )
    parser.add_argument(
        "--max_tune_samples", type=int,
        default=80,
        help="Maximum training samples to use during hyperparameter tuning"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Show per-sample evaluation results"
    )
    parser.add_argument(
        "--format_type", type=str,
        default="generation",
        choices=["generation", "chat"],
        help="Format type for prompt construction: 'generation' for simple concatenation, 'chat' for chat template"
    )
    parser.add_argument(
        "--enable_thinking", action="store_true",
        help="Enable thinking tokens in chat template format"
    )
    parser.add_argument(
        "--steering_token_position", type=int,
        default=-1,
        help="Token position for computing steering vectors (-1 for last token, 0 for first generated token, etc.)"
    )
    parser.add_argument(
        "--parallel_gpus", action="store_true",
        help="Enable parallel processing across multiple GPUs for rollout generation and evaluation"
    )
    parser.add_argument(
        "--grouped_normalization", action="store_true",
        help="Use per-question normalization before global averaging (One Question One Vote strategy)"
    )
    args = parser.parse_args()
    
    # Initialize distributed training if enabled
    # Check if running under torchrun or torch.distributed.launch
    local_rank = args.local_rank
    if local_rank == -1:
        local_rank = int(os.environ.get('LOCAL_RANK', -1))
    
    world_size = args.world_size
    if world_size == 1:
        world_size = int(os.environ.get('WORLD_SIZE', 1))
    
    use_distributed = args.parallel_gpus and local_rank >= 0 and world_size > 1
    
    if use_distributed:
        init_distributed(local_rank, world_size)
        torch.cuda.set_device(local_rank)
        num_available_gpus = world_size
    else:
        num_available_gpus = torch.cuda.device_count() if torch.cuda.is_available() and args.parallel_gpus else 1
    
    # Create output directory (only on main process)
    output_dir = Path(args.output_dir)
    if is_main_process():
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Print GPU info
    if use_distributed:
        print_rank0(f"🔥 Distributed training enabled. Using {num_available_gpus} GPUs.")
    elif args.parallel_gpus and torch.cuda.is_available():
        print_rank0(f"🔥 Multi-GPU mode enabled. Using {num_available_gpus} GPUs.")
    elif args.parallel_gpus and not torch.cuda.is_available():
        print_rank0("⚠️  Multi-GPU mode requested but CUDA not available. Using single GPU.")
        num_available_gpus = 1
    # Map layer scope string to enum
    scope_map = {
        "all": LayerScope.ALL,
        "first_n": LayerScope.FIRST_N,
        "last_n": LayerScope.LAST_N
    }
    # Initialize configurations
    rollout_config = RolloutConfig(
        num_rollouts=args.num_rollouts,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        use_reread_fallback=not args.no_reread,
        format_type=args.format_type,
        enable_thinking=args.enable_thinking
    )
    # We'll update intervention_config inside the loop if tuning is enabled
    base_intervention_config = InterventionConfig(
        layer_scope=scope_map[args.layer_scope],
        num_layers=args.num_layers,
        scaling_method=args.scaling,
        intervention_strength=args.strength,
        components=args.components.split(","),
        prefill_only=args.prefill_only,
        steering_token_position=args.steering_token_position,
        use_grouped_normalization=args.grouped_normalization
    )
    try:
        for data in args.datasets:
            args.dataset = data
            # Print configuration (only on main process)
            print_rank0("\n" + "="*60)
            print_rank0("🚀 ORBIT: ON-DISTRIBUTION ROLLOUT-BASED INTERVENTION")
            print_rank0("="*60)
            print_rank0(f"Model: {args.model}")
            print_rank0(
                f"Dataset: {args.dataset}, Tokens: {args.max_new_tokens}, Batch: {args.batch_size}")
            print_rank0(f"Baseline Evaluation: {'enabled' if args.eval_baseline else 'skipped'}")
            print_rank0(f"Rollouts: {args.num_rollouts} @ T={args.temperature}")
            print_rank0(f"Scaling: {args.scaling}, Strength: {args.strength}")
            print_rank0(f"Layer scope: {args.layer_scope} (n={args.num_layers})")
            print_rank0(f"Prefill only: {'enabled' if args.prefill_only else 'disabled'}")
            print_rank0(
                f"Re-read fallback: {'disabled' if args.no_reread else f'enabled (weight={args.reread_weight})'}")
            print_rank0(f"Format type: {args.format_type}")
            print_rank0(f"Steering token position: {args.steering_token_position}")
            print_rank0(f"Grouped normalization: {'enabled' if args.grouped_normalization else 'disabled'}")
            print_rank0(f"Seeds: {args.seeds}")
            print_rank0("="*60 + "\n")
            # Load model once (each process loads its own copy)
            print_rank0("📦 Loading model...")
        # Use the first seed for initial experiment config
        temp_exp_config = ExperimentConfig(
            model_name=args.model,
            dtype=args.dtype,
            rollout=rollout_config,
            intervention=base_intervention_config,
            train_samples=args.max_train,
            reread_weight=args.reread_weight,
            seed=args.seeds[0]
        )
        model = ModelWrapper(temp_exp_config)
        # Load dataset once
        print_rank0(f"\n📂 Loading dataset: {args.dataset}")
        loader = DatasetLoader(data_root=args.data_root)
        train_data, full_test_data = loader.load(
            args.dataset,
            max_train=args.max_train,
            max_test=args.max_test
        )
        all_seed_results = []
        best_overall_accuracy = -1.0
        best_seed_data = None
        for seed in args.seeds:
            print_rank0("\n" + "#"*60)
            print_rank0(f"🌟 RUNNING WITH SEED: {seed}")
            print_rank0("#"*60)
            # Set seed
            set_seed(seed)
            # Split test data into dev and test sets for hyperparameter tuning (if enabled)
            if args.tune_hyperparams and len(full_test_data) > 10:
                dev_size = max(5, int(len(full_test_data) * args.dev_ratio))
                test_size = len(full_test_data) - dev_size
                # Use fixed seed for reproducible split per seed run
                split_seed = seed + 1000
                random.seed(split_seed)
                indices = list(range(len(full_test_data)))
                random.shuffle(indices)  # Shuffle to make splits different across seeds
                dev_indices = indices[:dev_size]
                test_indices = indices[dev_size:dev_size + test_size]
                dev_data = [full_test_data[i] for i in dev_indices]
                test_data = [full_test_data[i] for i in test_indices]
                print_rank0(
                    f"Split test data: {len(dev_data)} dev, {len(test_data)} test (from {len(full_test_data)} total)")
            else:
                # No tuning or not enough samples, use all for test
                dev_data = full_test_data
                test_data = full_test_data
                print_rank0(f"Using all {len(test_data)} test samples")
            # Reset intervention config for each seed
            current_intervention_config = copy.deepcopy(base_intervention_config)
            # Hyperparameter tuning on dev set (if enabled)
            best_hyper_config = {
                'strength': current_intervention_config.intervention_strength,
                'num_layers': current_intervention_config.num_layers,
                'components': current_intervention_config.components
            }
            dev_results = []
            dev_baseline_result = None
            tune_steering_data = None  # Will store steering data from tuning if available
            tune_samples = 0  # Number of samples used in tuning
            if args.tune_hyperparams:
                print_rank0("\n🎯 Hyperparameter Tuning on Dev Set...")
                # Run baseline on dev set
                dev_baseline_result = evaluate_model(
                    model, dev_data,
                    intervention=None,
                    max_new_tokens=args.max_new_tokens,
                    batch_size=args.batch_size,
                    show_progress=True,
                    desc="Dev Baseline",
                    verbose=args.verbose,
                    num_gpus=num_available_gpus
                )
                print_rank0(f"Dev Baseline Accuracy: {dev_baseline_result.accuracy:.4f}")
                best_hyper_config, dev_results, tune_steering_data, tune_samples = tune_hyperparameters(
                    model, train_data, dev_data,
                    rollout_config, current_intervention_config,
                    args, seed, num_gpus=num_available_gpus
                )
                # Update configurations with best parameters
                current_intervention_config.intervention_strength = best_hyper_config['strength']
                current_intervention_config.num_layers = best_hyper_config['num_layers']
                current_intervention_config.components = best_hyper_config['components']
                print_rank0(f"🏆 Best hyperparameters: strength={best_hyper_config['strength']}, "
                      f"layers={best_hyper_config['num_layers']}, components={best_hyper_config['components']}")
            # Standard experiment on test set
            baseline_result = None
            if args.eval_baseline:
                print_rank0("\n📈 Baseline Evaluation...")
                baseline_result = evaluate_model(
                    model, test_data,
                    intervention=None,
                    max_samples=args.max_test,
                    max_new_tokens=args.max_new_tokens,
                    batch_size=args.batch_size,
                    desc="Baseline",
                    verbose=args.verbose,
                    num_gpus=num_available_gpus
                )
                print_rank0(f"Baseline Accuracy: {baseline_result.accuracy:.4f}")
            # Check if we can reuse steering vectors from tuning phase
            # Reuse if: tuning was done AND tuning used >= max_train samples
            can_reuse = (tune_steering_data is not None and
                         tune_samples >= (args.max_train or 100))
            if can_reuse:
                print_rank0("\n♻️ Reusing steering vectors from tuning phase...")
                steering_data = tune_steering_data
            else:
                print_rank0("\n🔧 Building Steering Vectors...")
                steering_data = build_steering_vectors(
                    model, train_data,
                    rollout_config, current_intervention_config,
                    reread_weight=args.reread_weight,
                    max_samples=args.max_train,
                    diff_batch_size=args.diff_batch_size,
                    show_progress=True,
                    format_type=args.format_type,
                    num_gpus=num_available_gpus
                )
            intervention = ActivationIntervention(
                model, current_intervention_config, steering_data["diff_result"]
            )
            print_rank0("\n📈 ORBIT Evaluation...")
            intervention_result = evaluate_model(
                model, test_data,
                intervention=intervention,
                max_samples=args.max_test,
                max_new_tokens=args.max_new_tokens,
                batch_size=args.batch_size,
                desc="ORBIT",
                verbose=args.verbose,
                num_gpus=num_available_gpus
            )
            print_rank0(f"ORBIT Accuracy: {intervention_result.accuracy:.4f}")
            delta = None
            if baseline_result:
                delta = intervention_result.accuracy - baseline_result.accuracy
                print_rank0(f"Improvement: {delta:+.4f}")
            current_run_results = {
                "seed": seed,
                "baseline": evaluation_result_to_dict(baseline_result) if baseline_result else None,
                "intervention": evaluation_result_to_dict(intervention_result),
                "delta": delta,
                "steering_stats": steering_data["stats"],
                "best_hyper_config": best_hyper_config,
                "dev_baseline": dev_baseline_result
            }
            all_seed_results.append(current_run_results)
            # Track best seed data for saving
            if intervention_result.accuracy > best_overall_accuracy:
                best_overall_accuracy = intervention_result.accuracy
                best_seed_data = {
                    "results": {
                        "baseline": evaluation_result_to_dict(baseline_result) if baseline_result else None,
                        "intervention": evaluation_result_to_dict(intervention_result),
                        "delta": delta,
                        "steering_stats": steering_data["stats"]
                    },
                    "hyperparameter_tuning": {
                        "best_config": best_hyper_config,
                        "dev_set_size": len(dev_data),
                        "test_set_size": len(test_data)
                    },
                    "seed": seed
                }
        # Calculate statistics
        accuracies = [r['intervention']['accuracy'] for r in all_seed_results]
        baseline_accuracies = [r['baseline']['accuracy']
                               for r in all_seed_results] if args.eval_baseline else []
        mean_acc = np.mean(accuracies)
        std_acc = np.std(accuracies)
        print_rank0("\n" + "="*60)
        print_rank0("📊 FINAL RESULTS SUMMARY")
        print_rank0("="*60)
        if args.eval_baseline:
            mean_base = np.mean(baseline_accuracies)
            std_base = np.std(baseline_accuracies)
            print_rank0(f"Base (Baseline) Accuracy: {mean_base:.4f} ± {std_base:.4f}")
        print_rank0(f"ORBIT Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
        if args.eval_baseline:
            print_rank0(f"Average Improvement: {mean_acc - mean_base:+.4f}")
        print_rank0("="*60)
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create model name for filename (simplify model path)
        model_name = args.model.split('/')[-1].replace('-', '_').replace('.', '_')
        train_size = args.max_train or 100  # Default to 100 if not specified

        result_file = output_dir / f"{args.dataset}_{model_name}_{train_size}_{timestamp}.json"
        final_output = {
            "config": {
                "model": args.model,
                "dataset": args.dataset,
                "max_new_tokens": args.max_new_tokens,
                "batch_size": args.batch_size,
                "eval_baseline": args.eval_baseline,
                "num_rollouts": args.num_rollouts,
                "temperature": args.temperature,
                "scaling": args.scaling,
                "strength": current_intervention_config.intervention_strength,
                "layer_scope": args.layer_scope,
                "num_layers": args.num_layers,
                "reread_weight": args.reread_weight,
                "format_type": args.format_type,
                "enable_thinking": args.enable_thinking,
                "steering_token_position": args.steering_token_position,
                "grouped_normalization": args.grouped_normalization,
                "seeds": args.seeds,
                "components": current_intervention_config.components,
                "prefill_only": args.prefill_only
            },
            "stats": {
                "mean_accuracy": mean_acc,
                "std_accuracy": std_acc,
                "all_accuracies": accuracies
            },
            "best_seed_run": best_seed_data,
            "steering_vectors": serialize_steering_vectors(steering_data)
        }
        if args.eval_baseline:
            final_output["stats"]["mean_baseline"] = np.mean(baseline_accuracies)
            final_output["stats"]["std_baseline"] = np.std(baseline_accuracies)
        if is_main_process():
            with open(result_file, "w") as f:
                json.dump(final_output, f, indent=2)
            print_rank0(f"\n💾 Results saved to: {result_file}")
        print_rank0("\n✅ Experiment completed!")
    finally:
        # Cleanup distributed training
        if use_distributed:
            cleanup_distributed()
        
if __name__ == "__main__":
    main()
