"""
Main entry point for ORBIT (On-distribution Rollout-based Behavioral Intervention Technique).

Implements two key improvements:
1. Rollout-based contrastive pair generation (ORC)
2. Continuous soft scaling (CSS)

Usage:
    python main.py --model meta-llama/Llama-3.1-8B-Instruct --dataset copa
    python main.py --model Qwen/Qwen3-7B-Instruct --dataset sst2
"""
import json
import torch
import argparse
import random
import numpy as np
import itertools
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from typing import List, Tuple, Dict, Optional

from config import (
    ExperimentConfig, 
    RolloutConfig, 
    InterventionConfig, 
    LayerScope
)
from models.wrapper import ModelWrapper
from steering.rollout import RolloutGenerator
from steering.diff_vector import ContinuousDiffCalculator
from steering.intervention import ActivationIntervention
from data.loader import DatasetLoader
from utils.metrics import Evaluator


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
    args
):
    """
    Tune hyperparameters on dev set to avoid overfitting to test set.

    Returns:
        tuple: (best_config, dev_results)
    """
    # Define hyperparameter search space
    param_grid = {
        'strength': [0.01,0.05,0.08,0.2, 0.5, 0.8, 1.0, 2.0],
    }

    # Generate all combinations dynamically
    keys, values = zip(*param_grid.items())
    all_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    # Limit combinations for efficiency
    max_combinations = 12

    # Randomly sample combinations if too many
    if len(all_combinations) > max_combinations:
        random.seed(args.seed + 2000)  # Different seed for hyperparameter sampling
        all_combinations = random.sample(all_combinations, max_combinations)

    print(f"Testing {len(all_combinations)} hyperparameter combinations on dev set...")

    best_score = 0.0
    best_config = None
    dev_results = []

    for i, config in enumerate(all_combinations):
        # Get current parameters, falling back to initial config if not in grid
        curr_strength = config.get('strength', intervention_config.intervention_strength)
        curr_layers = config.get('num_layers', intervention_config.num_layers)
        curr_components = config.get('components', intervention_config.components)

        print(f"\n🔧 Config {i+1}/{len(all_combinations)}: "
              f"strength={curr_strength}, layers={curr_layers}, "
              f"components={curr_components}")

        try:
            # Create config with current hyperparameters
            current_intervention_config = InterventionConfig(
                layer_scope=intervention_config.layer_scope,
                num_layers=curr_layers,
                scaling_method=intervention_config.scaling_method,
                intervention_strength=curr_strength,
                components=curr_components,
                prefill_only=intervention_config.prefill_only
            )

            # Build steering vectors with reduced samples for speed
            max_train_samples = min(args.max_train or 100, args.max_tune_samples)
            steering_data = build_steering_vectors(
                model, train_data,
                rollout_config, current_intervention_config,
                reread_weight=args.reread_weight,
                max_samples=max_train_samples,
                show_progress=False,
                diff_batch_size=args.diff_batch_size
            )

            # Create intervention
            intervention = ActivationIntervention(
                model, current_intervention_config, steering_data["diff_result"]
            )

            # Evaluate on dev set
            dev_result = evaluate_model(
                model, dev_data,
                intervention=intervention,
                max_samples=len(dev_data),  # Use all dev samples
                max_new_tokens=args.max_new_tokens,
                batch_size=args.batch_size,
                show_progress=False,
                desc=f"Dev {i+1}"
            )

            score = dev_result['accuracy']
            dev_results.append({
                'config': config,
                'accuracy': score,
                'correct': dev_result['correct'],
                'total': dev_result['total']
            })

            print(f"Dev accuracy: {score:.4f}")

            # Update best
            if score >= best_score:
                best_score = score
                best_config = {
                    'strength': curr_strength,
                    'num_layers': curr_layers,
                    'components': curr_components
                }

        except Exception as e:
            print(f"❌ Failed: {e}")
            dev_results.append({
                'config': config,
                'accuracy': 0.0,
                'correct': 0,
                'total': len(dev_data),
                'error': str(e)
            })

    print(f"\n✅ Tuning completed. Best score: {best_score:.4f}")
    return best_config, dev_results


def build_steering_vectors(
    model: ModelWrapper,
    train_data: List[Tuple[str, str, str]],
    rollout_config: RolloutConfig,
    intervention_config: InterventionConfig,
    reread_weight: float = 0.5,
    max_samples: int = 150,
    show_progress: bool = True,
    diff_batch_size: int = 16
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
    
    # Collect all contrastive pairs first
    all_pairs_data = []  # List of (question, positive, negative, used_reread)
    stats = {
        "total_samples": 0,
        "samples_with_rollout_correct": 0,
        "samples_with_reread": 0,
        "total_pairs": 0,
        "reread_pairs": 0,
        "skipped_samples": 0
    }
    
    samples = train_data[:max_samples]
    iterator = tqdm(samples, desc="Generating rollouts") if show_progress else samples
    
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
            batch_iter = tqdm(batch_iter, desc="Computing diffs (batch)")
        
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
                    tqdm.write(f"Warning: Batch diff failed: {e}")
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
                            tqdm.write(f"Warning: Single diff failed: {e2}")
    
    if not all_diffs:
        raise RuntimeError("No valid difference vectors computed. Check your data and model.")
    
    # Aggregate all diffs with re-read weighting
    diff_result = diff_calc.aggregate_diffs(
        all_diffs, 
        reread_flags,
        reread_weight=reread_weight
    )
    
    print("\n📊 Steering Vector Statistics:")
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
    max_new_tokens: int = 16,
    batch_size: int = 16,
    show_progress: bool = True,
    desc: str = "Evaluating"
) -> Dict:
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
    
    Returns:
        Dictionary with evaluation results.
    """
    evaluator = Evaluator(verbose=False)
    
    samples = test_data[:max_samples] if max_samples else test_data
    
    # Process in batches
    num_batches = (len(samples) + batch_size - 1) // batch_size
    iterator = range(0, len(samples), batch_size)
    if show_progress:
        iterator = tqdm(iterator, total=num_batches, desc=desc)
    
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
            
            for response, correct_ans in zip(responses, correct_answers):
                evaluator.evaluate_single(response, correct_ans)
            
        except Exception as e:
            if show_progress:
                tqdm.write(f"Warning: Batch evaluation failed: {e}")
    
    result = evaluator.get_result()
    return {
        "accuracy": result.accuracy,
        "correct": result.correct,
        "total": result.total
    }


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
        default="bfloat16",
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
        default=16,
        help="Batch size for diff computation (pairs per batch)"
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
        "--seed", type=int, 
        default=42,
        help="Random seed"
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
        default=0.3,
        help="Ratio of test data to use for dev set (0.0-1.0)"
    )
    parser.add_argument(
        "--max_tune_samples", type=int,
        default=80,
        help="Maximum training samples to use during hyperparameter tuning"
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
        max_new_tokens=args.max_new_tokens,
        use_reread_fallback=not args.no_reread
    )
    
    intervention_config = InterventionConfig(
        layer_scope=scope_map[args.layer_scope],
        num_layers=args.num_layers,
        scaling_method=args.scaling,
        intervention_strength=args.strength,
        components=args.components.split(","),
        prefill_only=args.prefill_only
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
    print(f"Dataset: {args.dataset}, Tokens: {args.max_new_tokens}, Batch: {args.batch_size}")
    print(f"Baseline Evaluation: {'enabled' if args.eval_baseline else 'skipped'}")
    print(f"Rollouts: {args.num_rollouts} @ T={args.temperature}")
    print(f"Scaling: {args.scaling}, Strength: {args.strength}")
    print(f"Layer scope: {args.layer_scope} (n={args.num_layers})")
    print(f"Prefill only: {'enabled' if args.prefill_only else 'disabled'}")
    print(f"Re-read fallback: {'disabled' if args.no_reread else f'enabled (weight={args.reread_weight})'}")
    print("="*60 + "\n")
    
    # Load model
    print("📦 Loading model...")
    model = ModelWrapper(experiment_config)
    
    # Load dataset
    print(f"\n📂 Loading dataset: {args.dataset}")
    loader = DatasetLoader(data_root=args.data_root)
    train_data, full_test_data = loader.load(
        args.dataset,
        max_train=args.max_train,
        max_test=args.max_test
    )

    # Split test data into dev and test sets for hyperparameter tuning (if enabled)
    if args.tune_hyperparams and len(full_test_data) > 10:
        dev_size = max(5, int(len(full_test_data) * args.dev_ratio))
        test_size = len(full_test_data) - dev_size

        # Use fixed seed for reproducible split
        split_seed = args.seed + 1000
        random.seed(split_seed)
        indices = list(range(len(full_test_data)))
        dev_indices = indices[:dev_size]
        test_indices = indices[dev_size:dev_size + test_size]
        dev_data = [full_test_data[i] for i in dev_indices]
        test_data = [full_test_data[i] for i in test_indices]

        print(f"Split test data: {len(dev_data)} dev, {len(test_data)} test (from {len(full_test_data)} total)")
    else:
        # No tuning or not enough samples, use all for test
        dev_data = full_test_data
        test_data = full_test_data
        print(f"Using all {len(test_data)} test samples (hyperparameter tuning: {'enabled' if args.tune_hyperparams else 'disabled'})")
    
    # Hyperparameter tuning on dev set (if enabled)
    best_config = {
        'strength': intervention_config.intervention_strength,
        'num_layers': intervention_config.num_layers,
        'components': intervention_config.components
    }
    dev_results = []
    dev_baseline_result = None

    if args.tune_hyperparams:
        print("\n" + "="*60)
        print("🎯 HYPERPARAMETER TUNING ON DEV SET")
        print("="*60)

        # Run baseline on dev set
        print("\n📊 Evaluating baseline on dev set...")
        dev_baseline_result = evaluate_model(
            model, dev_data,
            intervention=None,
            max_new_tokens=args.max_new_tokens,
            batch_size=args.batch_size,
            show_progress=True,
            desc="Dev Baseline"
        )
        print(f"Dev Baseline Accuracy: {dev_baseline_result['accuracy']:.4f}")

        best_config, dev_results = tune_hyperparameters(
            model, train_data, dev_data,
            rollout_config, intervention_config,
            args
        )

        # Update configurations with best parameters
        intervention_config.intervention_strength = best_config['strength']
        intervention_config.num_layers = best_config['num_layers']
        intervention_config.components = best_config['components']

        print(f"\n🏆 Best hyperparameters: strength={best_config['strength']}, "
              f"layers={best_config['num_layers']}, components={best_config['components']}")
    else:
        print("\n⏭️  Skipping hyperparameter tuning (use --tune_hyperparams to enable)")

    # Standard experiment on full test set
    baseline_result = None
    if args.eval_baseline:
        print("\n" + "="*60)
        print("📈 BASELINE EVALUATION")
        print("="*60)

        baseline_result = evaluate_model(
            model, test_data,
            intervention=None,
            max_samples=args.max_test,
            max_new_tokens=args.max_new_tokens,
            batch_size=args.batch_size,
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
        max_samples=args.max_train,
        diff_batch_size=args.diff_batch_size
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
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        desc="With intervention"
    )
    
    print(f"Intervention Accuracy: {intervention_result['accuracy']:.4f}")
    
    delta = None
    if baseline_result:
        delta = intervention_result['accuracy'] - baseline_result['accuracy']
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
                "max_new_tokens": args.max_new_tokens,
                "batch_size": args.batch_size,
                "eval_baseline": args.eval_baseline,
                "num_rollouts": args.num_rollouts,
                "temperature": args.temperature,
                "scaling": args.scaling,
                "strength": args.strength,
                "layer_scope": args.layer_scope,
                "num_layers": args.num_layers,
                "reread_weight": args.reread_weight,
                "seed": args.seed
            },
            "hyperparameter_tuning": {
                "dev_results": dev_results,
                "dev_baseline": dev_baseline_result,
                "best_config": best_config,
                "dev_set_size": len(dev_data),
                "test_set_size": len(test_data)
            },
            "results": results
        }, f, indent=2)
    
    print(f"\n💾 Results saved to: {result_file}")
    print("\n✅ Experiment completed!")


if __name__ == "__main__":
    main()
