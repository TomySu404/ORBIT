#!/bin/bash

# --- 基础配置 (参考 launch.json) ---
DATA_ROOT="/data4/xuanbo.su/ORBIT/data"
BATCH_SIZE=32
SADI_TOPK=6
NUM_ROLLOUTS=1
LAYER_SCOPE="all"

# --- 矩阵搜索空间 ---
# 1. 模型列表
MODELS=(
    "/data2/models/Qwen3-0.6B"
    "/data2/models/gemma3-it-1b"
    "/data2/models/Qwen3-8B"
)

# 2. 数据集列表
DATASETS=('sst2' 'sst5' 'mmlu' 'xnli' 'winogrande' 'truthfulqa')

# 3. 训练样本数列表
MAX_TRAIN_LIST=(100 1000)

# --- 输出根目录 ---
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_BASE="./results/sadi_matrix_$TIMESTAMP"
mkdir -p "$OUTPUT_BASE"

echo "Starting SADI matrix evaluation..."
echo "Models: ${MODELS[*]}"
echo "Datasets: ${DATASETS[*]}"
echo "Max Train Values: ${MAX_TRAIN_LIST[*]}"
echo "Output Directory: $OUTPUT_BASE"

# --- 嵌套循环执行 ---
for MODEL in "${MODELS[@]}"
do
    # 提取模型名称用于文件夹命名 (例如 Qwen3-8B)
    MODEL_NAME=$(basename "$MODEL")

    for DS in "${DATASETS[@]}"
    do
        for MT in "${MAX_TRAIN_LIST[@]}"
        do
            echo "=========================================================="
            echo "Model: $MODEL_NAME | Dataset: $DS | Max Train: $MT"
            echo "=========================================================="

            # 构造唯一的输出子目录路径
            # 格式: results/sadi_matrix_.../Qwen3-8B/sst2/train_100
            CUR_OUTPUT="$OUTPUT_BASE/$MODEL_NAME/$DS/train_$MT"
            mkdir -p "$CUR_OUTPUT"
            
            # 执行 main.py
            # 使用您 launch.json 中的核心参数
            python main.py \
                --model "$MODEL" \
                --datasets "$DS" \
                --data_root "$DATA_ROOT" \
                --batch_size "$BATCH_SIZE" \
                --sadi_topk "$SADI_TOPK" \
                --max_train "$MT" \
                --num_rollouts "$NUM_ROLLOUTS" \
                --layer_scope "$LAYER_SCOPE" \
                --sadi_pairs \
                --prefill_only \
                --sadi \
                --tune_hyperparams \
                --output_dir "$CUR_OUTPUT" \
                --format_type "generation" \
                2>&1 | tee "$CUR_OUTPUT/run.log"

            echo "Finished. Results saved in: $CUR_OUTPUT"
        done
    done
done

echo "=========================================================="
echo "All tasks in the matrix have been completed!"