#!/usr/bin/env bash

export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"
export hf_token=""

echo "Starting training ..."

python3 save_embeddings.py \
    --retriever_model_path ./retriever \
    --output_folder vector_database \
    --build_db

python3 inference_batch.py \
    --retriever_model_path ./retriever \
    --reranker_model_path ./reranker \
    --index_folder ./vector_database \
    --test_data_path "data/test_open.txt" \

