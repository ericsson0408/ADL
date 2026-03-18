#!/usr/bin/env bash

export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"

# Create log file with timestamp
LOG_FILE="retriever_training_$(date +%Y%m%d_%H%M%S).log"

echo "Starting retriever training with evaluation..."
echo "Log file: $LOG_FILE"

python3 retriever.py \
    --train_path "./data/train.txt" \
    --corpus_path "./data/corpus.txt" \
    --qrels_path "./data/qrels.txt" \
    --model_name "intfloat/multilingual-e5-small" \
    --output_dir "retriever" \
    --batch_size 64 \
    --epochs 5 \
    --lr 2e-5 \
    --max_seq_len 512 \
    --warmup_ratio 0.1 \
    --max_pos_per_query 8 \
    --eval_steps 500 \
    --use_evaluation 2>&1 | tee "$LOG_FILE"
    
echo "Training completed!"
echo "Log saved to: $LOG_FILE"
