#!/bin/bash
export CUDA_VISIBLE_DEVICES=7
# 基础配置
MODEL_BASE_PATH="/data2/models"
NUM_LAYERS=5
REREAD_WEIGHT=1
MAX_TUNE_SAMPLES=1000
BATCH_SIZE=32
FORMAT_TYPE="chat"

OUTPUT_DIR="./results_models"
mkdir -p "$OUTPUT_DIR"

# 初始默认值 (Baseline)
CURR_MT=100
CURR_NR=32
CURR_MNT=128
CURR_COMP="mlp_act"
CURR_SCOPE="all"
CURR_SCL="max_norm"
CURR_SADI=True
CURR_SADI_TOPK=5
CURR_SADI_SELECTION="pos"
CURR_SADI_MASK_SCOPE="global"

MODELS=("/data2/models/Qwen3-0.6B" "/data2/models/gemma-3-1b-it" "/data2/models/Qwen3-8B")
DATASETS=("sst2" "sst5" "mmlu" "truthfulqa" "winogrande" "xnli")
MT_OPTS=(100,1000)
SEEDS=(42 52)
NR_OPTS=(64)
# # 搜索空间 (注意：全部用空格分隔，不要用逗号！)
# MODELS=("/data2/models/Qwen3-0.6B")
# DATASETS=("sst2")
# MT_OPTS=(100 500 1000)
# NR_OPTS=(1 8 16 32 64)
# MNT_OPTS=(1 32 128 256)
# COMP_OPTS=("mlp" "attn" "mlp,attn" "attn_out" "mlp_act")
# SCOPE_OPTS=("first_n" "last_n" "all")
# SCL_OPTS=("max_norm" "softmax" "l2_norm")
# SEEDS=(22 42 52)

# 运行实验并返回准确率
run_and_get_acc() {
    # $1=MODEL, $2=DATASET, $3=MT, $4=NR, $5=MNT, $6=COMP, $7=SCOPE, $8=SCL
    # 运行 main.py，将输出重定向到 stderr（这样终端能看到进度）
    torchrun --master_port=29503 --nproc_per_node=2 main.py \
        --model "$1" \
        --dataset "$2" \
        --batch_size "$BATCH_SIZE" \
        --max_train "$3" \
        --num_rollouts "$4" \
        --max_new_tokens "$5" \
        --components "$6" \
        --layer_scope "$7" \
        --scaling "$8" \
        --num_layers "$NUM_LAYERS" \
        --max_tune_samples "$MAX_TUNE_SAMPLES" \
        --strength "0.2" \
        --tune_hyperparams \
        --grouped_normalization \
        --prefill_only \
        --output_dir "$OUTPUT_DIR" \
        --format_type "$FORMAT_TYPE" \
        --parallel_gpus \
        --seeds "${SEEDS[@]}" >&2
    
    # 检查结果文件并提取准确率（只输出数字到 stdout）
    latest_file=$(ls -t "$OUTPUT_DIR"/"$2"_*.json 2>/dev/null | head -n 1)
    if [ -z "$latest_file" ]; then
        echo "0.0"
    else
        python3 -c "import json; print(json.load(open('$latest_file'))['stats']['mean_accuracy'])" 2>/dev/null || echo "0.0"
    fi
}

# 浮点数比较 (替代 bc)
is_better() {
    python3 -c "print(1 if float('${1:-0}') >= float('${2:-0}') else 0)"
}

for MODEL in "${MODELS[@]}"; do
    for DATASET in "${DATASETS[@]}"; do
        if [ "$DATASET" == "gsm8k" ] || [ "$DATASET" == "math500" ]; then
            FORMAT_TYPE="chat"
        else
            FORMAT_TYPE="generation"
        fi
        echo "🌟 Starting Sequential Optimization for Model=$MODEL, Dataset=$DATASET, FORMAT_TYPE=$FORMAT_TYPE..."

        # 1. 优化 NUM_ROLLOUTS
        max_acc="0.0"; best_nr=$CURR_NR
        for val in "${NR_OPTS[@]}"; do
            echo "🔎 Testing NUM_ROLLOUTS=$val..."
            acc=$(run_and_get_acc "$MODEL" "$DATASET" "$CURR_MT" "$val" "$CURR_MNT" "$CURR_COMP" "$CURR_SCOPE" "$CURR_SCL")
            echo "   -> Accuracy: $acc"
            if [ "$(is_better "$acc" "$max_acc")" -eq 1 ]; then max_acc=$acc; best_nr=$val; fi
        done
        CURR_NR=$best_nr
        echo "✅ Best NUM_ROLLOUTS: $CURR_NR (Acc: $max_acc)"

        # 2. 优化 MAX_NEW_TOKENS
        max_acc="0.0"; best_mnt=$CURR_MNT
        for val in "${MNT_OPTS[@]}"; do
            echo "🔎 Testing MAX_NEW_TOKENS=$val..."
            acc=$(run_and_get_acc "$MODEL" "$DATASET" "$CURR_MT" "$CURR_NR" "$val" "$CURR_COMP" "$CURR_SCOPE" "$CURR_SCL")
            echo "   -> Accuracy: $acc"
            if [ "$(is_better "$acc" "$max_acc")" -eq 1 ]; then max_acc=$acc; best_mnt=$val; fi
        done
        CURR_MNT=$best_mnt
        echo "✅ Best MAX_NEW_TOKENS: $CURR_MNT (Acc: $max_acc)"

        # 3. 优化 COMPONENTS
        max_acc="0.0"; best_comp=$CURR_COMP
        for val in "${COMP_OPTS[@]}"; do
            echo "🔎 Testing COMPONENTS=$val..."
            acc=$(run_and_get_acc "$MODEL" "$DATASET" "$CURR_MT" "$CURR_NR" "$CURR_MNT" "$val" "$CURR_SCOPE" "$CURR_SCL")
            echo "   -> Accuracy: $acc"
            if [ "$(is_better "$acc" "$max_acc")" -eq 1 ]; then max_acc=$acc; best_comp=$val; fi
        done
        CURR_COMP=$best_comp
        echo "✅ Best COMPONENTS: $CURR_COMP (Acc: $max_acc)"

        # 4. 优化 LAYER_SCOPE
        max_acc="0.0"; best_scope=$CURR_SCOPE
        for val in "${SCOPE_OPTS[@]}"; do
            echo "🔎 Testing LAYER_SCOPE=$val..."
            acc=$(run_and_get_acc "$MODEL" "$DATASET" "$CURR_MT" "$CURR_NR" "$CURR_MNT" "$CURR_COMP" "$val" "$CURR_SCL")
            echo "   -> Accuracy: $acc"
            if [ "$(is_better "$acc" "$max_acc")" -eq 1 ]; then max_acc=$acc; best_scope=$val; fi
        done
        CURR_SCOPE=$best_scope
        echo "✅ Best LAYER_SCOPE: $CURR_SCOPE (Acc: $max_acc)"

        # 5. 优化 SCALING
        max_acc="0.0"; best_scl=$CURR_SCL
        for val in "${SCL_OPTS[@]}"; do
            echo "🔎 Testing SCALING=$val..."
            acc=$(run_and_get_acc "$MODEL" "$DATASET" "$CURR_MT" "$CURR_NR" "$CURR_MNT" "$CURR_COMP" "$CURR_SCOPE" "$val")
            echo "   -> Accuracy: $acc"
            if [ "$(is_better "$acc" "$max_acc")" -eq 1 ]; then max_acc=$acc; best_scl=$val; fi
        done
        CURR_SCL=$best_scl
        echo "✅ Best SCALING: $CURR_SCL (Acc: $max_acc)"
        echo ""
        echo "🏁 Final Best Config for $MODEL on $DATASET:"
        echo "   MAX_TRAIN=$CURR_MT, NUM_ROLLOUTS=$CURR_NR, MAX_NEW_TOKENS=$CURR_MNT"
        echo "   COMPONENTS=$CURR_COMP, LAYER_SCOPE=$CURR_SCOPE, SCALING=$CURR_SCL"
        echo "================================================"
    done
done

echo "All experiments finished!"