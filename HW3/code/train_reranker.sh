#!/usr/bin/env bash

export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"

echo "=========================================="
echo "Step 1: Generate hard negatives from retriever"
echo "=========================================="

python3 gen_hard_neg.py \
    --train_path "./data/train.txt" \
    --qrels_path "./data/qrels.txt" \
    --retriever_model_path "./retriever" \
    --index_folder "./vector_database" \
    --output_path "./data/train_with_hard_negs.jsonl" \
    --top_k 50

echo ""
echo "=========================================="
echo "Step 2: Train reranker with hard negatives"
echo "=========================================="

python3 reranker.py \
    --train_path "./data/train_with_hard_negs.jsonl" \
    --corpus_path "./data/corpus.txt" \
    --qrels_path "./data/qrels.txt" \
    --model_name "cross-encoder/ms-marco-MiniLM-L-12-v2" \
    --epochs 1 \
    --batch_size 32 \
    --grad_accum 2 \
    --max_negs 7 \
    --lr 2e-5 \
    --eval_steps 1000 \
    --warmup_ratio 0.1 \
    --dev_size 0.1 \
    --max_length 384 \
    --save_path "reranker" \
    --use_hard_negatives


