import torch
import json
import argparse
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from utils import get_prompt, get_bnb_config

def generate_predictions(
    model,
    tokenizer,
    data,
    output_path,
    max_new_tokens=256 # 您可以根據需求調整生成長度
):
    """
    使用模型生成預測並將其保存為指定的 JSON 格式。

    Args:
        model: 用於生成文本的 PEFT 模型。
        tokenizer: 用於編碼和解碼文本的分詞器。
        data (list): 包含 'id' 和 'instruction' 的字典列表。
        output_path (str): 儲存輸出結果的檔案路徑。
        max_new_tokens (int): 控制生成文本的最大長度。
    """
    model.eval()
    results = []
    
    # 使用 tqdm 顯示進度條
    for item in tqdm(data, desc="Generating Predictions"):
        instruction = item.get("instruction", "")
        item_id = item.get("id")

        if not instruction or not item_id:
            continue

        # 格式化提示
        prompt = get_prompt(instruction)
        
        # 將提示編碼並移至 GPU
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)

        with torch.no_grad():
            # 生成 token IDs
            # 參數可以微調以獲得更好的結果
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=True, # 使用採樣以增加多樣性
                top_p=0.9,
                temperature=0.4 # 較低的溫度使輸出更具確定性
            )

        # 只解碼新生成的部分，並跳過特殊符號
        # generated_ids[0] 的形狀是 [input_length + output_length]
        # 我們只取 input_length 後面的部分
        output_ids = generated_ids[0][inputs['input_ids'].shape[1]:]
        
        # 解碼為文字，並去除頭尾空白
        # skip_special_tokens=True 會自動移除 <s>, </s> 等符號
        output_text = tokenizer.decode(output_ids, skip_special_tokens=True).strip()

        # 將結果存入列表
        results.append({
            "id": item_id,
            "output": output_text
        })

    # 將結果寫入 JSON 檔案
    with open(output_path, 'w', encoding='utf-8') as f:
        # ensure_ascii=False 確保中文字符能正確寫入
        # indent=2 使 JSON 檔案格式化，易於閱讀
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ 推論完成！結果已儲存至 {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_model_path",
        type=str,
        default="Qwen/Qwen3-4B",
        help="基礎模型的路徑。",
    )
    parser.add_argument(
        "--peft_path",
        type=str,
        required=True,
        help="PEFT checkpoint 的儲存路徑。",
    )
    parser.add_argument(
        "--test_data_path",
        type=str,
        required=True,
        help="測試資料的路徑 (.json)。",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="輸出預測結果的檔案路徑 (.json)。",
    )
    args = parser.parse_args()

    # 載入模型和 tokenizer
    bnb_config = get_bnb_config()

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path)
    
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if tokenizer.bos_token_id is None:
        tokenizer.bos_token_id = tokenizer.eos_token_id

    # 載入 LoRA 適配器
    model = PeftModel.from_pretrained(model, args.peft_path)

    # 讀取測試資料
    with open(args.test_data_path, "r", encoding='utf-8') as f:
        data = json.load(f)

    # 執行生成
    generate_predictions(model, tokenizer, data, args.output_path)