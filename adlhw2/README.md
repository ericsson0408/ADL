# ADL HW2 inference Guide

This Python script is designed to perform Instruction Fine-Tuning on a large language model (LLM) using the QLoRA (Quantized Low-Rank Adaptation) technique.

## Setup

To get started, run the `download.sh` script. This will automatically download all the necessary components:

* PEFT checkpoints

Run:

    bash download.sh
## Inference

To execute the full inference, simply run the `run.sh` script. It will generate the final `output.json` file in the root directory.

    bash run.sh /path/model /path/adapter_checkpoint /path/public_test.json /path/output.json

Make sure to have `utils.py` in the same directory as  `get_bnb_config()` and `get_prompt()` are not defined in the main file. Additionally, the `inf.py` will be needed during the execution

## Training (Reproducibility)

If you wish to retrain the models from scratch, you can use the `train.sh` script (in source). This will initiate the training process.

Make sure to have `utils.py` in the same directory as  `get_bnb_config()` and `get_prompt()` are not defined in the main file.

Additionally, the `train.py` will be needed during the execution of `train.sh`.

**Prerequisites:** Before running the training script, please ensure that the necessary data files, `public_test.json` and `train.json` are available in the designated directory as train

    bash train.sh

## Acknowledgements & Implementation Details

* **Model**: This project utilizes the `Qwen/Qwen3-4B` model for training, as required by the Teaching Assistant (TA). `yentinglin/Llama-3.1-Taiwan-8B` was also trained and evaluated for bonus question.

* **Codebase**:
    * The `train.py` and `inf.py` scripts were developed based on the Instruction Tuning and QLoRA training guides provided in the TA powerpoint.

    * The `plot.py` script was coded by assistance of Gemini based on instructions provided to plot the model's perplexity and eval_loss.

    * The `pplNT.py` script was modified by the code `ppl.py` given from the TA to try zero_shot and few_shot without fine-tuning.

    * Assistance from Large Language Models (LLM) was leveraged to complete the implementation of these scripts.