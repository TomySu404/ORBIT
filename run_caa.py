import torch
import numpy as np
import random
from torch import nn
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

from config import ExperimentConfig, InterventionConfig, LayerScope, RolloutConfig
from models.wrapper import ModelWrapper
from data.loader import DatasetLoader
from steering.diff_vector import ContinuousDiffCalculator, DiffVectorResult
from steering.intervention import ActivationIntervention
from utils.metrics import Evaluator

# /y104p32_data4/models
class Config:
    MODEL_NAME = "/data2/models/Qwen3-8B" 
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DTYPE = "bfloat16" 
    
    SEEDS = [42,52]

    TARGET_LAYERS = [10, 15, 20]
    
    COMPONENT = "block_output" 
    
    TOKEN_POS = -1
    
    MULTIPLIERS = [1, 2, 5 ,10]

    DEV_RATIO = 0.2

    DATASETS = ["gsm8k"]
    MAX_SAMPLES = 100
    VECTOR_BATCH_SIZE = 20
    DATA_ROOT = "/data4/xuanbo.su/ORBIT/data"

class CAAModelWrapper(ModelWrapper):
    def get_layer_name(self, layer_idx: int, component: str) -> str:
        if component == "block_output":
            return self.layer_pattern["layers_template"].format(layer_idx=layer_idx)
        return super().get_layer_name(layer_idx, component)

class CAAIntervention(ActivationIntervention):
    @staticmethod
    def _intervention_hook(module, input, output, intervention, config_prefill_only, initial_seq_len_ref, token_position=-1):
        if isinstance(output, tuple):
            hidden = output[0]
            rest = output[1:]
            is_tuple = True
        else:
            hidden = output
            rest = None
            is_tuple = False

        seq_len = hidden.shape[1]

        if seq_len > 1: return output

        interv = intervention.to(hidden.device).to(hidden.dtype)
        hidden = hidden + interv

        if is_tuple: return (hidden,) + rest
        return hidden



def get_raw_vector(model: CAAModelWrapper, dataset: str) -> Dict[str, torch.Tensor]:
    print(f"[-] Calculating steering vector on TRAIN set: {dataset}")
    loader = DatasetLoader(data_root=Config.DATA_ROOT)

    train_data, _ = loader.load(dataset, split="train", max_train=Config.MAX_SAMPLES)
    
    if not train_data:
        raise ValueError(f"No training data found for dataset: {dataset}")
        
    cfg = InterventionConfig(
        layer_scope=LayerScope.CUSTOM,
        custom_layers=Config.TARGET_LAYERS, 
        components=[Config.COMPONENT],
        steering_token_position=Config.TOKEN_POS
    )
    
    diff_calc = ContinuousDiffCalculator(model, cfg, format_type="chat")
    
    all_diffs = []
    batch_size = Config.VECTOR_BATCH_SIZE
    
    for i in range(0, len(train_data), batch_size):
        batch = train_data[i : i + batch_size]
        qs, pos, neg = zip(*batch)
        print(f"    Processing batch {i//batch_size + 1}/{(len(train_data)-1)//batch_size + 1} ({len(batch)} samples)")
        batch_diffs = diff_calc.compute_batch_pair_diffs(list(qs), list(pos), list(neg))
        all_diffs.extend(batch_diffs)
        
    agg_result = diff_calc.aggregate_diffs(all_diffs)
    
    return agg_result.diff_vectors

def calculate_scale_factors(raw_vectors_map: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, float]:
    print("\n>>> Calculating Scale Factors...")
    if not raw_vectors_map: return {}
    
    dataset_norms = {}
    
    for dataset, layer_dict in raw_vectors_map.items():
        layer_norms = [v.norm(p=2).item() for v in layer_dict.values()]
        avg_layer_norm = sum(layer_norms) / len(layer_norms) if layer_norms else 0.0
        dataset_norms[dataset] = avg_layer_norm

    mean_norm = sum(dataset_norms.values()) / len(dataset_norms)
    print(f"    Global Mean Norm: {mean_norm:.4f}")
    
    return {d: (mean_norm / n if n > 1e-6 else 1.0) for d, n in dataset_norms.items()}

def create_caa_intervention(
    model: CAAModelWrapper, 
    layer_vectors: Dict[str, torch.Tensor], 
    multiplier: float
) -> ActivationIntervention:
    config = InterventionConfig(
        layer_scope=LayerScope.CUSTOM,
        custom_layers=Config.TARGET_LAYERS,
        components=[Config.COMPONENT],
        intervention_strength=multiplier,
        prefill_only=False
    )
    
    fake_data = DiffVectorResult(
        diff_vectors={k: torch.empty(0) for k in layer_vectors}, 
        scaling_weights=layer_vectors 
    )
    
    return CAAIntervention(model, config, fake_data)



def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    print(f">>> Loading Model: {Config.MODEL_NAME}")

    rollout_config = RolloutConfig(
        format_type="chat",  
        max_new_tokens=128,
        temperature=1.0 
    )

    config = ExperimentConfig(
        model_name=Config.MODEL_NAME, 
        device=Config.DEVICE, 
        dtype=Config.DTYPE,
        rollout=rollout_config 
    )
    
    model = CAAModelWrapper(config)

    # Dictionary to store accuracy results: {dataset: {multiplier: [acc_seed1, acc_seed2, ...]}}
    all_accuracies = defaultdict(lambda: defaultdict(list))

    for seed_idx, seed in enumerate(Config.SEEDS):
        print(f"\n" + "="*50)
        print(f">>> Running Experiment with Seed {seed} ({seed_idx + 1}/{len(Config.SEEDS)})")
        print("="*50)
        
        set_seed(seed)

        print("\n>>> Collecting Steering Vectors (Residual Stream)...")
        raw_vectors = {}
        for dataset in Config.DATASETS:
            try:
                raw_vectors[dataset] = get_raw_vector(model, dataset)
            except Exception as e:
                print(f"Error calculating vector for {dataset}: {e}")

        scale_factors = calculate_scale_factors(raw_vectors)

        for dataset in Config.DATASETS:
            if dataset not in raw_vectors: continue
                
            print(f"\n>>> Running CAA Evaluation on Dataset: {dataset} (Seed {seed})")
            
            loader = DatasetLoader(data_root=Config.DATA_ROOT)
            _, full_test_data = loader.load(dataset, split="test")
            
            if not full_test_data:
                print(f"    Warning: No test data found for {dataset}")
                continue

            # Split test data into dev and test sets
            dev_size = max(5, int(len(full_test_data) * Config.DEV_RATIO))
            if len(full_test_data) <= dev_size + 5:
                # Too few samples, use all for both (not ideal but better than nothing)
                dev_data = full_test_data
                test_data = full_test_data
                print(f"    Warning: Too few samples for split, using all {len(full_test_data)} for both dev and test")
            else:
                # Use fixed seed for reproducible split per seed run
                split_seed = seed + 1000
                random.seed(split_seed)
                indices = list(range(len(full_test_data)))
                random.shuffle(indices)
                dev_indices = indices[:dev_size]
                test_indices = indices[dev_size:]
                dev_data = [full_test_data[i] for i in dev_indices]
                test_data = [full_test_data[i] for i in test_indices]
                print(f"    Split: {len(dev_data)} dev, {len(test_data)} test")
            
            # Helper to evaluate on a dataset
            def evaluate_on_split(data, multiplier_val, desc="Eval"):
                qs_local, refs_local = zip(*[(x[0], x[1]) for x in data])
                raw_vec_dict = raw_vectors[dataset]
                scale = scale_factors[dataset]
                normalized_vec_dict = {k: v * scale for k, v in raw_vec_dict.items()}
                
                intervention = create_caa_intervention(model, normalized_vec_dict, multiplier_val)
                evaluator = Evaluator(verbose=False)
                
                resps = []
                eval_batch_size = Config.VECTOR_BATCH_SIZE
                for i in range(0, len(qs_local), eval_batch_size):
                    batch_qs = list(qs_local)[i : i + eval_batch_size]
                    batch_resps = intervention.generate_with_intervention(
                        batch_qs, 
                        max_new_tokens=1024, 
                        token_position=-1,
                        do_sample=False  
                    )
                    resps.extend(batch_resps)
                
                correct = sum(evaluator.evaluate_single(r, ref, record=False) for r, ref in zip(resps, refs_local))
                accuracy = correct / len(refs_local)
                return accuracy

            # 1. Search for best multiplier on DEV set
            print(f"    [-] Searching best multiplier on DEV set...")
            best_m = Config.MULTIPLIERS[0]
            best_dev_acc = -1.0
            
            for m in Config.MULTIPLIERS:
                dev_acc = evaluate_on_split(dev_data, m, desc="Dev")
                print(f"        Multiplier {m:2} | Dev Accuracy: {dev_acc:.4%}")
                if dev_acc > best_dev_acc:
                    best_dev_acc = dev_acc
                    best_m = m
            
            print(f"    [+] Best Multiplier found: {best_m} (Dev Acc: {best_dev_acc:.4%})")
            
            # 2. Evaluate best multiplier on TEST set
            test_acc = evaluate_on_split(test_data, best_m, desc="Test")
            print(f"    [!] Final Test Accuracy (m={best_m}): {test_acc:.4%}")
            
            all_accuracies[dataset][best_m].append(test_acc)
            # We also store it in a way that we can calculate overall mean regardless of which multiplier was chosen
            if "best_test_accuracies" not in all_accuracies[dataset]:
                all_accuracies[dataset]["best_test_accuracies"] = []
            all_accuracies[dataset]["best_test_accuracies"].append(test_acc)

    # Final summary
    print("\n" + "#"*50)
    print(">>> FINAL RESULTS SUMMARY (Mean ± Std)")
    print("#"*50)
    
    for dataset in Config.DATASETS:
        if dataset not in all_accuracies or "best_test_accuracies" not in all_accuracies[dataset]:
            continue
            
        print(f"\nDataset: {dataset}")
        accs = all_accuracies[dataset]["best_test_accuracies"]
        if not accs:
            continue
        
        mean_acc = np.mean(accs)
        std_acc = np.std(accs)
        print(f"  Best Dev-selected Multipliers | Test Accuracy: {mean_acc:.4%} ± {std_acc:.4%}")

if __name__ == "__main__":
    main()