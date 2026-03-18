#!/bin/bash

export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"
echo "======================================="
echo "Starting training MC by trainMC.py ..."
echo "======================================="

#bert-base-chinese
#hfl/chinese-macbert-base
#hfl/chinese-pert-base

python3 trainMC.py \
    --model_name_or_path hfl/chinese-lert-large\
    --tokenizer_name hfl/chinese-lert-large\
    --train_file ./data/train.json \
    --validation_file ./data/valid.json \
    --context_file ./data/context.json \
    --max_seq_length 512 \
    --output_dir ./MC_lertlarge_model \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --per_device_eval_batch_size 2 \
    --num_train_epochs 3
echo "======================================="
echo "model saved to MC.model"
echo "======================================="

echo "======================================="
echo "Starting training QA by trainQA.py ..."
echo "======================================="
#bert-base-chinese
#hfl/chinese-macbert-base
#hfl/chinese-macbert-large
#hfl/chinese-roberta-wwm-ext-large
#hfl/chinese-lert-large


python3 trainQA.py \
  --model_name_or_path hfl/chinese-lert-large \
  --train_file data/train.json \
  --validation_file data/valid.json \
  --context_file data/context.json \
  --output_dir QA_long_model \
  --num_train_epochs 3 \
  --learning_rate 5e-5 \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps 2\
  --seed 42\
  --max_seq_length 512 \
  --n_best_size 5 \
  --with_tracking \
  --report_to tensorboard \
  --pad_to_max_length

echo "======================================="
echo "model saved to QA.model"
echo "======================================="



