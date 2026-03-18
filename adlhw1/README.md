# ADL HW1 interence Guide test1

This project provides a complete pipeline for a two-stage inference process involving a Multiple-Choice (MC) model followed by a Question-Answering (QA) model.

## Setup

To get started, run the `download.sh` script. This will automatically download all the necessary components:
* Source code(including inferMC.py, inferQA.py)
* Required data(content.json, test.json ONLY)
* Pre-trained models for both the MC and QA tasks

Run:

    bash download.sh
## Inference

To execute the full inference pipeline, simply run the `run.sh` script. It will handle both the MC and QA steps sequentially and generate the final `prediction.csv` file in the root directory.

    bash run.sh /path/context.json /path/test.json /path/prediction.csv

## Training (Reproducibility)

If you wish to retrain the models from scratch, you can use the `train.sh` script. This will initiate the training process for both the MC and QA models.

The `utils_qa.py` is not needed as it is merged into `trainQA.py`.
The `em_curve.png` and `loss_curve.png` will also be drawn by `trainQA.py` during the execution of `train.sh`.

**Prerequisites:** Before running the training script, please ensure that the necessary data files, `train.json` and `valid.json`, are available in the designated directory.

    bash train.sh

## Acknowledgements & Implementation Details

* **Model**: This project utilizes the `hfl/chinese-lert-large` model for both MC and QA training, as permitted by the Teaching Assistant (TA).

* **Codebase**:
    * The `trainMC.py` and `infMC.py` scripts were developed with reference to the `run_swag_no_trainer.py` script provided by the TA.
    * The `trainQA.py` and `infQA.py` scripts were developed with reference to the `run_qa_no_trainer.py` script provided by the TA.
    * Assistance from Large Language Models (LLM) was leveraged to complete the implementation of these scripts.