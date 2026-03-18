#!/bin/bash


#/home/guest/r14922159/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c
#./adapter_checkpoint
#./public_test.json
#./output.json

#./run.sh /home/guest/r14922159/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c ./adapter_checkpoint ./public_test.json ./output.json

# 當任何指令失敗時，立即停止腳本執行
set -e

# 檢查是否提供了四個必要的參數
if [ "$#" -ne 4 ]; then
    echo "錯誤：需要提供四個參數。"
    echo "用法: bash $0 <model_checkpoint_path> <adapter_checkpoint_path> <input_file_path> <output_file_path>"
    exit 1
fi

# 將命令列參數分配給具名變數，增加可讀性
MODEL_CHECKPOINT_PATH=${1}
ADAPTER_CHECKPOINT_PATH=${2}
INPUT_FILE_PATH=${3}
OUTPUT_FILE_PATH=${4}

echo "===== 開始執行推論（Inference） ====="
echo "模型路徑: ${MODEL_CHECKPOINT_PATH}"
echo "Adapter 路徑: ${ADAPTER_CHECKPOINT_PATH}"
echo "輸入檔案: ${INPUT_FILE_PATH}"
echo "輸出檔案: ${OUTPUT_FILE_PATH}"

# 執行您的 Python 推論腳本
# 注意：請將 'your_inference_script.py' 換成您實際的腳本名稱
# 同時，請確保您的 Python 腳本能接收 --base_model_path, --adapter_path, --input_file, 和 --output_file 這些參數
python3 $SCRIPT_DIR/Source/inf.py \
    --base_model_path "${MODEL_CHECKPOINT_PATH}" \
    --peft_path "${ADAPTER_CHECKPOINT_PATH}" \
    --test_data_path "${INPUT_FILE_PATH}" \
    --output_path "${OUTPUT_FILE_PATH}"

echo -e "\n✅ 推論完成！預測結果已儲存至 ${OUTPUT_FILE_PATH}"

