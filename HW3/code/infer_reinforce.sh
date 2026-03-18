#!/usr/bin/env bash

export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"
export HUGGING_FACE_HUB_TOKEN=""

echo "Starting training ..."

python3 infer_rl.py \
    --retriever_model_path ./retriever \
    --reranker_model_path ./reranker \
    --index_folder ./vector_database \
    --test_data_path "data/test_open.txt" \
    --data_folder ./data \
    --passage_file "corpus.txt" \
    --index_file "passage_index.faiss" \
    --sqlite_file "passage_store.db" \
    --qrels_path "./data/qrels.txt" \
    --rl_agent_path "rag_rl_agent.zip" \
    --generator_model "Qwen/Qwen3-1.7B" \
    --result_file_name "result_rl.json"

