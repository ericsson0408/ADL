# predict.py
import argparse
import os
import torch
import json
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from accelerate import Accelerator
from transformers import (
    AutoModelForMultipleChoice,
    AutoTokenizer,
    default_data_collator,
)
from itertools import chain

def parse_args():
    parser = argparse.ArgumentParser(description="Run prediction on a multiple choice task with a trained model.")
    

    parser.add_argument(
        "--model_name_or_path",  
        type=str,
        required=True,
        help="Path to the trained model directory.",
    )
    parser.add_argument(
        "--test_file",
        type=str,
        required=True,
        help="A json file containing the test data (questions and paragraph indices)."
    )
    # --- NEW: 新增 context_file 參數 ---
    parser.add_argument(
        "--context_file",
        type=str,
        required=True,
        help="A json file containing the context paragraphs."
    )
    parser.add_argument(
        "--test_output",  # 從 output_file 改名
        type=str,
        required=True,
        help="The path to save the prediction results as a JSON file."
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=512,  # 建議為上下文任務使用較大的長度
        help="The maximum total input sequence length after tokenization.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    # Initialize accelerator
    accelerator = Accelerator()

    # --- 1. Load Model and Tokenizer ---
    print(f"Loading model from '{args.model_name_or_path}'...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModelForMultipleChoice.from_pretrained(args.model_name_or_path)
    
    # --- ADDED: 增加 token embeddings resize 的好習慣 ---
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))


    # --- 2. Load and Preprocess Data ---
    print(f"Loading test data from '{args.test_file}'...")
    print(f"Loading context data from '{args.context_file}'...")

    # --- 同時讀取 test_file 和 context_file ---
    raw_datasets = load_dataset("json", data_files={"test": args.test_file})
    with open(args.context_file, 'r', encoding='utf-8') as f:
        context = json.load(f)

    padding = "max_length"

    #==========直接弄的跟trainMC.py一樣（除了label）=======
    def preprocess_function(examples):
        question = [[question] * 4 for question in examples['question']]
        option = [[context[i] for i in options] for options in examples['paragraphs']]
        #labels = [examples['paragraphs'][i].index(examples['relevant'][i]) for i in range(len(examples['id']))]

        # Flatten out
        question = list(chain(*question))
        option = list(chain(*option))

        # Tokenize
        tokenized_examples = tokenizer(
            question,
            option,
            max_length=args.max_seq_length,
            padding=padding,
            truncation=True,
        )
        # Un-flatten
        tokenized_inputs = {k: [v[i : i + 4] for i in range(0, len(v), 4)] for k, v in tokenized_examples.items()}
        #tokenized_inputs["labels"] = labels
        return tokenized_inputs

    with accelerator.main_process_first():
        processed_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            remove_columns=raw_datasets["test"].column_names
        )

    test_dataset = processed_datasets["test"]

    # --- 3. Create DataLoader ---
    data_collator = default_data_collator
    test_dataloader = DataLoader(
        test_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size
    )

    model, test_dataloader = accelerator.prepare(model, test_dataloader)

    # --- 4. Run Prediction ---
    model.eval()
    all_predictions = []

    print("Running predictions...")
    for batch in tqdm(test_dataloader, desc="Predicting"):
        if "labels" in batch:
            batch.pop("labels")

        with torch.no_grad():
            outputs = model(**batch)

        predictions = outputs.logits.argmax(dim=-1)
        gathered_predictions = accelerator.gather(predictions)
        all_predictions.extend(gathered_predictions.cpu().numpy())

    accelerator.wait_for_everyone()

    # --- 5. Save processed data ---
    if accelerator.is_main_process:
        print(f"Saving predictions to '{args.test_output}'...")

        # 讀取原始的 test_file 以便修改它
        with open(args.test_file, 'r', encoding='utf-8') as f:
            result_json = json.load(f)
        
        if len(all_predictions) > len(result_json):
            all_predictions = all_predictions[:len(result_json)]
        
        assert len(result_json) == len(all_predictions), "預測數量與資料點數量不匹配！"

        # 在每筆資料中添加 'relevant' 欄位
        for i, prediction_index in enumerate(all_predictions):
            # 取得當前問題對應的段落索引列表
            paragraph_indices = result_json[i]['paragraphs']
            # 取得模型選擇的那個段落索引
            relevant_paragraph_id = paragraph_indices[prediction_index]
            # 將結果加入到字典中
            result_json[i]['relevant'] = int(relevant_paragraph_id)

        # 將修改後的完整資料結構寫入檔案
        with open(args.test_output, 'w', encoding='utf-8') as f:
            json.dump(result_json, f, ensure_ascii=False, indent=2)
            
        print("Prediction complete!")

if __name__ == "__main__":
    main()