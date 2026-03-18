#!/bin/bash

#./run.sh ./data/test.json ./data/context.json ./output.csv
# 檢查是否提供了三個必要的參數
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <path_to_context.json> <path_to_test.json> <path_to_output.csv>"
    exit 1
fi

# 將傳入的參數賦予變數，增加可讀性
CONTEXT_FILE=$1
TEST_FILE=$2
OUTPUT_FILE=$3

# 設定環境變數
export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"

# 設定 MC 模型預測的中間檔路徑
MC_PREDICTIONS_FILE="./predictions.json"

echo "======================================="
echo "Starting inference MC by inferMC.py ..."
echo "Input context: ${CONTEXT_FILE}"
echo "Input test file: ${TEST_FILE}"
echo "======================================="
python3 Source/inferMC.py \
    --model_name_or_path ./MC_lertlarge_model \
    --context_file "${CONTEXT_FILE}" \
    --test_file "${TEST_FILE}" \
    --test_output "${MC_PREDICTIONS_FILE}"

echo "======================================="
echo "Inference MC finished. Intermediate file saved to ${MC_PREDICTIONS_FILE}"
echo "======================================="
echo "Starting inference QA by inferQA.py ..."
echo "Input intermediate file: ${MC_PREDICTIONS_FILE}"
echo "Output prediction file: ${OUTPUT_FILE}"
echo "======================================="

# 注意：這裡假設您的 inferQA.py 接受一個 --output_file 參數來指定輸出路徑
# 如果您的參數名稱不同（例如 --prediction_file），請修改下面的 --output_file
python3 Source/inferQA.py \
  --model_name_or_path ./QA_lert_model \
  --context_file "${CONTEXT_FILE}" \
  --test_file "${MC_PREDICTIONS_FILE}" \
  --output_file "${OUTPUT_FILE}" \
  --max_seq_length 512 \
  --doc_stride 128 \
  --per_device_eval_batch_size 8 \
  --overwrite_cache

echo "======================================="
echo "Inference QA finished. Final predictions saved to ${OUTPUT_FILE}"
echo "======================================="