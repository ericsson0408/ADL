#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json, math, argparse, random
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import torch
from torch.utils.data import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    set_seed,
)

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)

from utils import get_prompt, get_bnb_config


# ----------------------------
# Data utilities
# ----------------------------

class JsonSupervisedDataset(Dataset):
    """
    Expecting a JSON array file with items like:
      { "instruction": "...", "output": "..." }
    """
    def __init__(self,
                 data: List[Dict[str, Any]],
                 tokenizer: AutoTokenizer,
                 max_length: int):
        self.data = data
        self.tok = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def _build_ids_and_labels(self, instruction: str, answer: str):
        # 使用與 ppl.py 完全一致的格式
        prompt_text = get_prompt(instruction)
        answer_text = (answer or "").rstrip() + self.tok.eos_token

        # 分別 tokenize，不加特殊 token
        prompt_ids = self.tok(prompt_text, add_special_tokens=False)["input_ids"]
        answer_ids = self.tok(answer_text, add_special_tokens=False)["input_ids"]

        # 手動加 BOS token（與 ppl.py 一致）
        if self.tok.bos_token_id is not None:
            input_ids = [self.tok.bos_token_id] + prompt_ids + answer_ids
            # 只對 answer 部分計算 loss
            labels = [-100] * (1 + len(prompt_ids)) + answer_ids
        else:
            input_ids = prompt_ids + answer_ids
            labels = [-100] * len(prompt_ids) + answer_ids

        # 長度控制：優先保留完整答案
        if len(input_ids) > self.max_length:
            # 從左側截斷（保留更多答案）
            overflow = len(input_ids) - self.max_length
            input_ids = input_ids[overflow:]
            labels = labels[overflow:]

        # 確保不超過 max_length
        input_ids = input_ids[:self.max_length]
        labels = labels[:self.max_length]

        return input_ids, labels

    def __getitem__(self, idx: int):
        ex = self.data[idx]
        instr = ex.get("instruction") or ex.get("input") or ""
        ans = ex.get("output") or ex.get("response") or ""
        input_ids, labels = self._build_ids_and_labels(instr, ans)
        return {"input_ids": input_ids, "labels": labels}


@dataclass
class DataCollatorForCausalLMWithPadding:
    tokenizer: AutoTokenizer
    pad_to_multiple_of: Optional[int] = 8

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        max_len = max(len(f["input_ids"]) for f in features)
        if self.pad_to_multiple_of is not None:
            if max_len % self.pad_to_multiple_of != 0:
                max_len = ((max_len // self.pad_to_multiple_of) + 1) * self.pad_to_multiple_of

        input_ids, attention_mask, labels = [], [], []
        pad_id = self.tokenizer.pad_token_id

        for f in features:
            ids = f["input_ids"]
            lbs = f["labels"]
            pad_len = max_len - len(ids)
            input_ids.append(ids + [pad_id] * pad_len)
            attention_mask.append([1] * len(ids) + [0] * pad_len)
            labels.append(lbs + [-100] * pad_len)

        batch = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
        return batch


# ----------------------------
# Model / Training utilities
# ----------------------------

def build_model_and_tokenizer(base_model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        use_fast=True,
    )
    # 確保 pad_token 設置
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # QLoRA 4-bit 設定
    if not torch.cuda.is_available():
        raise EnvironmentError(
            "QLoRA training requires a CUDA-enabled GPU, but no GPU was detected."
        )

    device_map = {"": torch.cuda.current_device()}
    bnb_config = get_bnb_config()

    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        quantization_config=bnb_config,
        device_map=device_map,
    )
    model.config.use_cache = False

    # Align model/generation configs with tokenizer to avoid repeated warnings
    model.config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.generation_config.bos_token_id = tokenizer.bos_token_id

    # 準備 k-bit 訓練 + 啟用 gradient checkpointing
    model = prepare_model_for_kbit_training(model)
    model.gradient_checkpointing_enable()

    return model, tokenizer


def attach_lora(model,
                r: int = 16,
                lora_alpha: int = 32,
                lora_dropout: float = 0.1,
                bias: str = "none",
                task_type: str = "CAUSAL_LM"):
    # 針對 Qwen 架構的目標模組
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"]
    config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias=bias,
        task_type=task_type,
    )
    model = get_peft_model(model, config)
    try:
        model.enable_input_require_grads()
    except Exception:
        pass
    model.print_trainable_parameters()
    return model


def load_json_list(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
        assert isinstance(data, list), "Expect a JSON array."
        return data


# ----------------------------
# Main
# ----------------------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_path", type=str, default="Qwen/Qwen3-4B")
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--val_file", type=str, default=None)
    parser.add_argument("--val_ratio", type=float, default=0.02)
    parser.add_argument("--output_dir", type=str, default="./adapter")

    # sequence / batch
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)

    # training schedule
    parser.add_argument("--num_train_epochs", type=float, default=3.0)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--weight_decay", type=float, default=0.01)

    # logging / eval / save
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--eval_steps", type=int, default=50)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--save_total_limit", type=int, default=3)

    # LoRA
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.1)

    # misc
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    set_seed(args.seed)

    # --- model / tokenizer ---
    model, tokenizer = build_model_and_tokenizer(args.base_model_path)

    # --- LoRA ---
    model = attach_lora(
        model,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )

    # --- dataset ---
    train_data_all = load_json_list(args.train_file)

    if args.val_file:
        val_data = load_json_list(args.val_file)
    else:
        random.shuffle(train_data_all)
        cut = max(1, int(len(train_data_all) * (1.0 - args.val_ratio)))
        val_data = train_data_all[cut:]
        train_data_all = train_data_all[:cut]

    print(f"Training samples: {len(train_data_all)}, Validation samples: {len(val_data)}")

    train_dataset = JsonSupervisedDataset(train_data_all, tokenizer, args.max_length)
    eval_dataset = JsonSupervisedDataset(val_data, tokenizer, args.max_length)

    # --- collator ---
    data_collator = DataCollatorForCausalLMWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8)

    # --- training args ---
    use_bf16 = torch.cuda.is_bf16_supported()
    use_fp16 = not use_bf16

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        lr_scheduler_type="cosine",
        bf16=use_bf16,
        fp16=use_fp16,
        optim="paged_adamw_32bit",
        report_to="none",
        run_name="qlora_chinese_translation",
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        dataloader_num_workers=2,
        ddp_find_unused_parameters=False,
    )

    # --- trainer ---
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    # Debug: 查看第一筆訓練樣本
    first_item = train_dataset[0]
    first_decoded = tokenizer.decode(first_item["input_ids"], skip_special_tokens=False)
    print("\n[DEBUG] First training sample:\n", first_decoded[:300], "\n")

    # --- train ---
    trainer.train()
    
    # --- final eval ---
    eval_metrics = trainer.evaluate()
    print("\n[Final Eval] metrics:", eval_metrics)
    if "eval_loss" in eval_metrics:
        ppl = math.exp(eval_metrics["eval_loss"])
        print(f"[Final Eval] Perplexity = {ppl:.4f}")

    # --- save ---
    os.makedirs(args.output_dir, exist_ok=True)
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print(f"\n[Done] Adapter saved to: {args.output_dir}\n")


if __name__ == "__main__":
    main()
