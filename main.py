"""
Main entry point for ORBIT (On-distribution Rollout-based Behavioral Intervention Technique).

Implements three key improvements:
1. Rollout-based contrastive pair generation (ORC)
2. Continuous soft scaling (CSS)
3. Structural layer-wise ablation

Usage:
    python main.py --model meta-llama/Llama-3.1-8B-Instruct --dataset copa
    python main.py --model Qwen/Qwen2.5-7B-Instruct --dataset sst2 --ablation
"""
import os
import sys
import json
import torch
import argparse
import random
import numpy as np
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from typing import List, Tuple, Dict, Optional

from config import (
    ExperimentConfig, 
    RolloutConfig, 
    InterventionConfig, 
    LayerScope,
    ModelType
)
from models.wrapper import ModelWrapper
from steering.rollout import RolloutGenerator
from steering.diff_vector import ContinuousDiffCalculator, DiffVectorResult
from steering.intervention import ActivationIntervention
from data.loader import DatasetLoader
from utils.metrics import Evaluator, compare_results


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_steering_vectors(
    model: ModelWrapper,
    train_data: List[Tuple[str, str, str]],
    rollout_config: RolloutConfig,
    intervention_config: InterventionConfig,
    reread_weight: float = 0.5,
    max_samples: int = 150,
    show_progress: bool = True
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
    
    Returns:
        Dictionary containing:
        - diff_result: DiffVectorResult with aggregated vectors
        - stats: Statistics about the build process
        - layer_indices: List of layer indices being intervened
    """
    rollout_gen = RolloutGenerator(model, rollout_config)
    diff_calc = ContinuousDiffCalculator(model, intervention_config)
    
    all_diffs = []
    reread_flags = []
    stats = {
        "total_samples": 0,
        "samples_with_rollout_correct": 0,
        "samples_with_reread": 0,
        "total_pairs": 0,
        "reread_pairs": 0,
        "skipped_samples": 0
    }
    
    samples = train_data[:max_samples]
    iterator = tqdm(samples, desc="Building steering vectors") if show_progress else samples
    
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
        
        # Compute diff for each contrastive pair
        for pair in result.contrastive_pairs:
            try:
                diff = diff_calc.compute_pair_diff(
                    question,
                    pair.positive,
                    pair.negative,
                    components=intervention_config.components
                )
                all_diffs.append(diff)
                reread_flags.append(pair.used_reread)
                
                stats["total_pairs"] += 1
                if pair.used_reread:
                    stats["reread_pairs"] += 1
                    
            except Exception as e:
                if show_progress:
                    tqdm.write(f"Warning: Failed to compute diff: {e}")
    
    if not all_diffs:
        raise RuntimeError("No valid difference vectors computed. Check your data and model.")
    
    # Aggregate all diffs with re-read weighting
    diff_result = diff_calc.aggregate_diffs(
        all_diffs, 
        reread_flags,
        reread_weight=reread_weight
    )
    
    print(f"\n📊 Steering Vector Statistics:")
    print(f"   Total samples processed: {stats['total_samples']}")
    print(f"   Samples with rollout correct: {stats['samples_with_rollout_correct']}")
    print(f"   Samples using re-read: {stats['samples_with_reread']}")
    print(f"   Skipped samples: {stats['skipped_samples']}")
    print(f"   Total contrastive pairs: {stats['total_pairs']}")
    print(f"   Re-read pairs: {stats['reread_pairs']}")
    print(f"   Layers intervened: {diff_calc.layer_indices}")
    
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
    show_progress: bool = True,
    desc: str = "Evaluating"
) -> Dict:
    """
    Evaluate model on test data with optional intervention.
    
    Args:
        model: Model wrapper instance.
        test_data: List of (question, correct_answer, wrong_answer) tuples.
        intervention: Optional intervention handler.
        max_samples: Maximum test samples (None for all).
        show_progress: Whether to show progress bar.
        desc: Description for progress bar.
    
    Returns:
        Dictionary with evaluation results.
    """
    evaluator = Evaluator(verbose=False)
    
    samples = test_data[:max_samples] if max_samples else test_data
    iterator = tqdm(samples, desc=desc) if show_progress else samples
    
    for question, correct_ans, wrong_ans in iterator:
        try:
            if intervention:
                response = intervention.generate_with_intervention(
                    question,
                    max_new_tokens=16,
                    do_sample=False
                )
            else:
                response = model.generate(
                    question,
                    max_new_tokens=16,
                    do_sample=False
                )[0]
            
            evaluator.evaluate_single(response, correct_ans)
            
        except Exception as e:
            if show_progress:
                tqdm.write(f"Warning: Evaluation failed: {e}")
    
    result = evaluator.get_result()
    return {
        "accuracy": result.accuracy,
        "correct": result.correct,
        "total": result.total
    }


def run_ablation_study(
    model: ModelWrapper,
    train_data: List[Tuple[str, str, str]],
    test_data: List[Tuple[str, str, str]],
    rollout_config: RolloutConfig,
    base_config: InterventionConfig,
    reread_weight: float = 0.5,
    max_train: int = 150,
    max_test: Optional[int] = None
) -> Dict[str, Dict]:
    """
    Run layer-wise ablation study.
    
    Tests intervention effectiveness on:
    - First N layers (shallow processing)
    - Last N layers (deep processing)
    - All layers
    
    This helps understand where the steering vectors are most effective.
    
    Args:
        model: Model wrapper.
        train_data: Training data.
        test_data: Test data.
        rollout_config: Rollout configuration.
        base_config: Base intervention configuration.
        reread_weight: Weight for re-read samples.
        max_train: Maximum training samples.
        max_test: Maximum test samples.
    
    Returns:
        Dictionary mapping scope names to results.
    """
    results = {}
    
    # Baseline evaluation (no intervention)
    print("\n" + "="*60)
    print("📈 BASELINE (No Intervention)")
    print("="*60)
    
    baseline_result = evaluate_model(
        model, test_data, 
        intervention=None,
        max_samples=max_test,
        desc="Baseline evaluation"
    )
    results["baseline"] = baseline_result
    print(f"Accuracy: {baseline_result['accuracy']:.4f}")
    
    # Define ablation scopes
    ablation_scopes = [
        (LayerScope.FIRST_N, "first_5_layers", "🔵 FIRST 5 LAYERS"),
        (LayerScope.LAST_N, "last_5_layers", "🔴 LAST 5 LAYERS"),
        (LayerScope.ALL, "all_layers", "🟢 ALL LAYERS"),
    ]
    
    for scope, scope_key, scope_name in ablation_scopes:
        print("\n" + "="*60)
        print(f"{scope_name}")
        print("="*60)
        
        # Create config for this ablation
        ablation_config = InterventionConfig(
            layer_scope=scope,
            num_layers=base_config.num_layers,
            scaling_method=base_config.scaling_method,
            intervention_strength=base_config.intervention_strength,
            components=base_config.components
        )
        
        # Build steering vectors for this scope
        steering_data = build_steering_vectors(
            model, train_data,
            rollout_config, ablation_config,
            reread_weight=reread_weight,
            max_samples=max_train
        )
        
        # Create intervention handler
        intervention = ActivationIntervention(
            model, ablation_config, steering_data["diff_result"]
        )
        
        # Evaluate
        result = evaluate_model(
            model, test_data,
            intervention=intervention,
            max_samples=max_test,
            desc=f"Evaluating {scope_key}"
        )
        
        result["delta"] = result["accuracy"] - baseline_result["accuracy"]
        results[scope_key] = result
        
        print(f"Accuracy: {result['accuracy']:.4f} (Δ = {result['delta']:+.4f})")
    
    # Print summary
    print("\n" + "="*60)
    print("📊 ABLATION STUDY SUMMARY")
    print("="*60)
    print(f"{'Scope':<20} {'Accuracy':>10} {'Δ':>10}")
    print("-"*40)
    print(f"{'Baseline':<20} {results['baseline']['accuracy']:>10.4f} {'---':>10}")
    for scope, scope_key, _ in ablation_scopes:
        r = results[scope_key]
        print(f"{scope_key:<20} {r['accuracy']:>10.4f} {r['delta']:>+10.4f}")
    print("="*60)
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="ORBIT: On-distribution Rollout-based Behavioral Intervention Technique"
    )
    
    # Model arguments
    parser.add_argument(
        "--model", type=str, 
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="HuggingFace model name or path"
    )
    parser.add_argument(
        "--dtype", type=str, 
        default="float16",
        choices=["float16", "bfloat16", "float32"],
        help="Model precision"
    )
    
    # Dataset arguments
    parser.add_argument(
        "--dataset", type=str, 
        default="copa",
        help="Dataset name (copa, sst2, boolq, etc.)"
    )
    parser.add_argument(
        "--data_root", type=str,
        default="./data",
        help="Root directory for datasets"
    )
    parser.add_argument(
        "--max_train", type=int,
        default=150,
        help="Maximum training samples"
    )
    parser.add_argument(
        "--max_test", type=int,
        default=None,
        help="Maximum test samples (None for all)"
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
    
    # Intervention arguments
    parser.add_argument(
        "--strength", type=float, 
        default=1.0,
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
        default="all",
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
    
    # Experiment arguments
    parser.add_argument(
        "--ablation", action="store_true",
        help="Run layer-wise ablation study"
    )
    parser.add_argument(
        "--seed", type=int, 
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--output_dir", type=str,
        default="./results",
        help="Directory to save results"
    )
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
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
        use_reread_fallback=not args.no_reread
    )
    
    intervention_config = InterventionConfig(
        layer_scope=scope_map[args.layer_scope],
        num_layers=args.num_layers,
        scaling_method=args.scaling,
        intervention_strength=args.strength,
        components=args.components.split(",")
    )
    
    experiment_config = ExperimentConfig(
        model_name=args.model,
        dtype=args.dtype,
        rollout=rollout_config,
        intervention=intervention_config,
        train_samples=args.max_train,
        reread_weight=args.reread_weight,
        seed=args.seed
    )
    
    # Print configuration
    print("\n" + "="*60)
    print("🚀 ORBIT: ON-DISTRIBUTION ROLLOUT-BASED INTERVENTION")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Rollouts: {args.num_rollouts} @ T={args.temperature}")
    print(f"Scaling: {args.scaling}, Strength: {args.strength}")
    print(f"Layer scope: {args.layer_scope} (n={args.num_layers})")
    print(f"Re-read fallback: {'disabled' if args.no_reread else f'enabled (weight={args.reread_weight})'}")
    print("="*60 + "\n")
    
    # Load model
    print("📦 Loading model...")
    model = ModelWrapper(experiment_config)
    
    # Load dataset
    print(f"\n📂 Loading dataset: {args.dataset}")
    loader = DatasetLoader(data_root=args.data_root)
    train_data, test_data = loader.load(
        args.dataset,
        max_train=args.max_train,
        max_test=args.max_test
    )
    
    if args.ablation:
        # Run ablation study
        results = run_ablation_study(
            model, train_data, test_data,
            rollout_config, intervention_config,
            reread_weight=args.reread_weight,
            max_train=args.max_train,
            max_test=args.max_test
        )
    else:
        # Standard experiment
        print("\n" + "="*60)
        print("📈 BASELINE EVALUATION")
        print("="*60)
        
        baseline_result = evaluate_model(
            model, test_data,
            intervention=None,
            max_samples=args.max_test,
            desc="Baseline"
        )
        print(f"Baseline Accuracy: {baseline_result['accuracy']:.4f}")
        
        print("\n" + "="*60)
        print("🔧 BUILDING STEERING VECTORS")
        print("="*60)
        
        steering_data = build_steering_vectors(
            model, train_data,
            rollout_config, intervention_config,
            reread_weight=args.reread_weight,
            max_samples=args.max_train
        )
        
        intervention = ActivationIntervention(
            model, intervention_config, steering_data["diff_result"]
        )
        
        print("\n" + "="*60)
        print("📈 INTERVENTION EVALUATION")
        print("="*60)
        
        intervention_result = evaluate_model(
            model, test_data,
            intervention=intervention,
            max_samples=args.max_test,
            desc="With intervention"
        )
        
        delta = intervention_result['accuracy'] - baseline_result['accuracy']
        print(f"Intervention Accuracy: {intervention_result['accuracy']:.4f}")
        print(f"Improvement: {delta:+.4f}")
        
        results = {
            "baseline": baseline_result,
            "intervention": intervention_result,
            "delta": delta,
            "steering_stats": steering_data["stats"]
        }
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = output_dir / f"{args.dataset}_{timestamp}.json"
    
    with open(result_file, "w") as f:
        json.dump({
            "config": {
                "model": args.model,
                "dataset": args.dataset,
                "num_rollouts": args.num_rollouts,
                "temperature": args.temperature,
                "scaling": args.scaling,
                "strength": args.strength,
                "layer_scope": args.layer_scope,
                "num_layers": args.num_layers,
                "reread_weight": args.reread_weight,
                "seed": args.seed
            },
            "results": results
        }, f, indent=2)
    
    print(f"\n💾 Results saved to: {result_file}")
    print("\n✅ Experiment completed!")


if __name__ == "__main__":
    main()
