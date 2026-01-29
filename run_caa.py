import torch
from torch import nn
from typing import Dict, List, Tuple, Optional

from config import ExperimentConfig, InterventionConfig, LayerScope, RolloutConfig
from models.wrapper import ModelWrapper
from data.loader import DatasetLoader
from steering.diff_vector import ContinuousDiffCalculator, DiffVectorResult
from steering.intervention import ActivationIntervention
from utils.metrics import Evaluator


class Config:
    MODEL_NAME = "/data2/models/Qwen3-0.6B" 
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DTYPE = "bfloat16" 
    

    TARGET_LAYERS = [10, 15, 20]
    
    COMPONENT = "block_output" 
    
    TOKEN_POS = -2 
    
    MULTIPLIERS = [-0.5, -0.1, 0, 0.1, 0.5]

    DATASETS = ["gsm8k"]
    MAX_SAMPLES = 100
    DATA_ROOT = "./data"

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
        
    qs, pos, neg = zip(*train_data)

    cfg = InterventionConfig(
        layer_scope=LayerScope.CUSTOM,
        custom_layers=Config.TARGET_LAYERS, 
        components=[Config.COMPONENT],
        steering_token_position=Config.TOKEN_POS
    )
    
    diff_calc = ContinuousDiffCalculator(model, cfg)
    all_diffs = diff_calc.compute_batch_pair_diffs(list(qs), list(pos), list(neg))
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



def main():
    print(f">>> Loading Model: {Config.MODEL_NAME}")

    rollout_config = RolloutConfig(
        format_type="chat",  
        max_new_tokens=200,
        temperature=1.0 
    )

    config = ExperimentConfig(
        model_name=Config.MODEL_NAME, 
        device=Config.DEVICE, 
        dtype=Config.DTYPE,
        rollout=rollout_config 
    )
    
    model = CAAModelWrapper(config)


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
            
        print(f"\n>>> Running CAA Evaluation on Dataset: {dataset}")
        
        loader = DatasetLoader(data_root=Config.DATA_ROOT)
        _, test_data = loader.load(dataset, split="test")
        
        if not test_data:
            print(f"    Warning: No test data found for {dataset}")
            continue
            
        qs, refs = zip(*[(x[0], x[1]) for x in test_data])
        
        raw_vec_dict = raw_vectors[dataset]
        scale = scale_factors[dataset]
        
        normalized_vec_dict = {
            k: v * scale for k, v in raw_vec_dict.items()
        }

        evaluator = Evaluator(verbose=False) 

        for m in Config.MULTIPLIERS:

            intervention = create_caa_intervention(model, normalized_vec_dict, m)
            

            resps = intervention.generate_with_intervention(
                list(qs), 
                max_new_tokens=200, 
                token_position=-1,
                do_sample=False  
            )
            
            correct = sum(evaluator.evaluate_single(r, ref, record=False) for r, ref in zip(resps, refs))
            print(f"[-] Multiplier {m} | Accuracy: {correct/len(refs):.1%}")

if __name__ == "__main__":
    main()