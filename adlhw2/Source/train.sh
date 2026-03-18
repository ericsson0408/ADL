#!/usr/bin/env bash

export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Starting training ..."

python3 "$SCRIPT_DIR/try3.py" \
    --base_model_path Qwen/Qwen3-4B \
    --train_file "$SCRIPT_DIR/train.json" \
    --val_file "$SCRIPT_DIR/public_test.json" \
    --output_dir "$SCRIPT_DIR/adapter" \
    --max_length 512 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --num_train_epochs 3 \
    --learning_rate 5e-5 \
    --warmup_ratio 0.05 \
    --weight_decay 0.001 \
    --lora_r 16 --lora_alpha 32 --lora_dropout 0.2 \
    --logging_steps 20 --eval_steps 100 --save_steps 100 \
    --save_total_limit 20

echo "Finish training"
