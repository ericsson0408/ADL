#!/usr/bin/env python
import argparse
import json
import logging
import os
import collections
from typing import Optional

import datasets
import numpy as np
import pandas as pd
import torch
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger  # Ensure this import is here
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    DataCollatorWithPadding,
)

# Use the logger from accelerate which understands 'main_process_only'
logger = get_logger(__name__)


# =====================================================================================
# Content from utils_qa.py is merged below
# =====================================================================================

def postprocess_qa_predictions(
    examples,
    features,
    predictions: tuple[np.ndarray, np.ndarray],
    version_2_with_negative: bool = False,
    n_best_size: int = 20,
    max_answer_length: int = 30,
    null_score_diff_threshold: float = 0.0,
    output_dir: Optional[str] = None,
    prefix: Optional[str] = None,
    log_level: Optional[int] = logging.WARNING,
):
    """
    Post-processes the predictions of a question-answering model to convert them to answers that are substrings of the
    original contexts. This is the base postprocessing functions for models that only return start and end logits.
    """
    if len(predictions) != 2:
        raise ValueError("`predictions` should be a tuple with two elements (start_logits, end_logits).")
    all_start_logits, all_end_logits = predictions

    if len(predictions[0]) != len(features):
        raise ValueError(f"Got {len(predictions[0])} predictions and {len(features)} features.")

    # Build a map example to its corresponding features.
    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    features_per_example = collections.defaultdict(list)
    for i, feature in enumerate(features):
        features_per_example[example_id_to_index[feature["example_id"]]].append(i)

    # The dictionaries we have to fill.
    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    if version_2_with_negative:
        scores_diff_json = collections.OrderedDict()

    # Logging.
    logger.setLevel(log_level)
    logger.info(f"Post-processing {len(examples)} example predictions split into {len(features)} features.")

    # Let's loop over all the examples!
    for example_index, example in enumerate(tqdm(examples)):
        # Those are the indices of the features associated to the current example.
        feature_indices = features_per_example[example_index]

        min_null_prediction = None
        prelim_predictions = []

        # Looping through all the features associated to the current example.
        for feature_index in feature_indices:
            # We grab the predictions of the model for this feature.
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]
            # This is what will allow us to map some the positions in our logits to span of texts in the original
            # context.
            offset_mapping = features[feature_index]["offset_mapping"]
            # Optional `token_is_max_context`, if provided we will remove answers that do not have the maximum context
            # available in the current feature.
            token_is_max_context = features[feature_index].get("token_is_max_context", None)

            # Update minimum null prediction.
            feature_null_score = start_logits[0] + end_logits[0]
            if min_null_prediction is None or min_null_prediction["score"] > feature_null_score:
                min_null_prediction = {
                    "offsets": (0, 0),
                    "score": feature_null_score,
                    "start_logit": start_logits[0],
                    "end_logit": end_logits[0],
                }

            # Go through all possibilities for the `n_best_size` greater start and end logits.
            start_indexes = np.argsort(start_logits)[-1 : -n_best_size - 1 : -1].tolist()
            end_indexes = np.argsort(end_logits)[-1 : -n_best_size - 1 : -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    if (
                        start_index >= len(offset_mapping)
                        or end_index >= len(offset_mapping)
                        or offset_mapping[start_index] is None
                        or len(offset_mapping[start_index]) < 2
                        or offset_mapping[end_index] is None
                        or len(offset_mapping[end_index]) < 2
                    ):
                        continue
                    if end_index < start_index or end_index - start_index + 1 > max_answer_length:
                        continue
                    if token_is_max_context is not None and not token_is_max_context.get(str(start_index), False):
                        continue

                    prelim_predictions.append(
                        {
                            "offsets": (offset_mapping[start_index][0], offset_mapping[end_index][1]),
                            "score": start_logits[start_index] + end_logits[end_index],
                            "start_logit": start_logits[start_index],
                            "end_logit": end_logits[end_index],
                        }
                    )
        if version_2_with_negative and min_null_prediction is not None:
            prelim_predictions.append(min_null_prediction)
            null_score = min_null_prediction["score"]

        predictions = sorted(prelim_predictions, key=lambda x: x["score"], reverse=True)[:n_best_size]

        if (
            version_2_with_negative
            and min_null_prediction is not None
            and not any(p["offsets"] == (0, 0) for p in predictions)
        ):
            predictions.append(min_null_prediction)

        context = example["context"]
        for pred in predictions:
            offsets = pred.pop("offsets")
            pred["text"] = context[offsets[0] : offsets[1]]

        if len(predictions) == 0 or (len(predictions) == 1 and predictions[0]["text"] == ""):
            predictions.insert(0, {"text": "empty", "start_logit": 0.0, "end_logit": 0.0, "score": 0.0})

        scores = np.array([pred.pop("score") for pred in predictions])
        exp_scores = np.exp(scores - np.max(scores))
        probs = exp_scores / exp_scores.sum()

        for prob, pred in zip(probs, predictions):
            pred["probability"] = prob

        if not version_2_with_negative:
            all_predictions[example["id"]] = predictions[0]["text"]
        else:
            i = 0
            while predictions[i]["text"] == "":
                i += 1
            best_non_null_pred = predictions[i]
            score_diff = null_score - best_non_null_pred["start_logit"] - best_non_null_pred["end_logit"]
            scores_diff_json[example["id"]] = float(score_diff)
            if score_diff > null_score_diff_threshold:
                all_predictions[example["id"]] = ""
            else:
                all_predictions[example["id"]] = best_non_null_pred["text"]

        all_nbest_json[example["id"]] = [
            {k: (float(v) if isinstance(v, (np.float16, np.float32, np.float64)) else v) for k, v in pred.items()}
            for pred in predictions
        ]

    if output_dir is not None:
        if not os.path.isdir(output_dir):
            raise OSError(f"{output_dir} is not a directory.")

        prediction_file = os.path.join(
            output_dir, "predictions.json" if prefix is None else f"{prefix}_predictions.json"
        )
        nbest_file = os.path.join(
            output_dir, "nbest_predictions.json" if prefix is None else f"{prefix}_nbest_predictions.json"
        )
        if version_2_with_negative:
            null_odds_file = os.path.join(
                output_dir, "null_odds.json" if prefix is None else f"{prefix}_null_odds.json"
            )

        logger.info(f"Saving predictions to {prediction_file}.")
        with open(prediction_file, "w") as writer:
            writer.write(json.dumps(all_predictions, indent=4) + "\n")
        logger.info(f"Saving nbest_preds to {nbest_file}.")
        with open(nbest_file, "w") as writer:
            writer.write(json.dumps(all_nbest_json, indent=4) + "\n")
        if version_2_with_negative:
            logger.info(f"Saving null_odds to {null_odds_file}.")
            with open(null_odds_file, "w") as writer:
                writer.write(json.dumps(scores_diff_json, indent=4) + "\n")

    return all_predictions

# =====================================================================================
# Original inferQA.py content starts here
# =====================================================================================

def parse_args():
    """Parses command-line arguments for the prediction script."""
    parser = argparse.ArgumentParser(description="Run QA predictions on a trained model.")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help="Path to the pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--test_file",
        type=str,
        required=True,
        help="A JSON file containing the test data."
    )
    parser.add_argument(
        "--context_file",
        type=str,
        required=True,
        help="A JSON file containing the context paragraphs."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="predictions.csv",
        help="Where to store the final predictions in CSV format."
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=384,
        help="The maximum total input sequence length after tokenization.",
    )
    parser.add_argument(
        "--doc_stride",
        type=int,
        default=128,
        help="When splitting a long document, how much stride to take between chunks.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--overwrite_cache",
        action="store_true",
        help="Overwrite the cached training and evaluation sets."
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=1,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--n_best_size",
        type=int,
        default=20,
        help="The total number of n-best predictions to generate."
    )
    parser.add_argument(
        "--max_answer_length",
        type=int,
        default=30,
        help="The maximum length of an answer that can be generated.",
    )

    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    # Initialize accelerator
    accelerator = Accelerator()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output_file)
    if output_dir and accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # Load context from the external context file
    with open(args.context_file, "r", encoding="utf-8") as f:
        CONTEXT = json.load(f)

    # Load dataset
    data_files = {"test": args.test_file}
    extension = args.test_file.split(".")[-1]
    raw_datasets = load_dataset(extension, data_files=data_files)

    # Load tokenizer and model
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    model = AutoModelForQuestionAnswering.from_pretrained(args.model_name_or_path, config=config)

    model.to(accelerator.device)

    # Preprocessing
    column_names = raw_datasets["test"].column_names
    pad_on_right = tokenizer.padding_side == "right"
    max_seq_length = min(args.max_seq_length, tokenizer.model_max_length)

    def prepare_prediction_features(examples):
        question = examples['question']
        context = [CONTEXT[idx] for idx in examples['relevant']]

        tokenized_examples = tokenizer(
            question if pad_on_right else context,
            context if pad_on_right else question,
            truncation="only_second" if pad_on_right else "only_first",
            max_length=max_seq_length,
            stride=args.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        tokenized_examples["example_id"] = []
        for i in range(len(tokenized_examples["input_ids"])):
            sample_index = sample_mapping[i]
            original_id = examples["id"][sample_index]
            tokenized_examples["example_id"].append(original_id)

            sequence_ids = tokenized_examples.sequence_ids(i)
            context_index = 1 if pad_on_right else 0

            new_offsets = []
            original_offsets = tokenized_examples["offset_mapping"][i]
            for k, offset in enumerate(original_offsets):
                if sequence_ids[k] == context_index:
                    new_offsets.append(offset)
                else:
                    new_offsets.append(None)
            tokenized_examples["offset_mapping"][i] = new_offsets

        return tokenized_examples

    predict_examples = raw_datasets["test"]
    with accelerator.main_process_first():
        predict_dataset = predict_examples.map(
            prepare_prediction_features,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not args.overwrite_cache,
            desc="Running tokenizer on prediction dataset",
        )

    # DataLoader
    data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=(8 if accelerator.mixed_precision != 'no' else None))
    predict_dataset_for_model = predict_dataset.remove_columns(["example_id", "offset_mapping"])
    predict_dataloader = DataLoader(
        predict_dataset_for_model, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size
    )

    model, predict_dataloader = accelerator.prepare(model, predict_dataloader)

    # Prediction
    logger.info("***** Running Prediction *****")
    logger.info(f"  Num examples = {len(predict_dataset)}")
    logger.info(f"  Batch size = {args.per_device_eval_batch_size}")

    all_start_logits, all_end_logits = [], []
    model.eval()

    for batch in tqdm(predict_dataloader, desc="Predicting"):
        with torch.no_grad():
            outputs = model(**batch)
            start_logits = outputs.start_logits
            end_logits = outputs.end_logits

            start_logits = accelerator.pad_across_processes(start_logits, dim=1, pad_index=-100)
            end_logits = accelerator.pad_across_processes(end_logits, dim=1, pad_index=-100)

            all_start_logits.append(accelerator.gather_for_metrics(start_logits).cpu().numpy())
            all_end_logits.append(accelerator.gather_for_metrics(end_logits).cpu().numpy())

    start_logits_concat = np.concatenate(all_start_logits, axis=0)
    end_logits_concat = np.concatenate(all_end_logits, axis=0)

    start_logits_concat = start_logits_concat[:len(predict_dataset)]
    end_logits_concat = end_logits_concat[:len(predict_dataset)]

    outputs_numpy = (start_logits_concat, end_logits_concat)

    # Post-processing
    def add_context_to_examples(examples):
        examples["context"] = [CONTEXT[rel_id] for rel_id in examples["relevant"]]
        return examples

    predict_examples = predict_examples.map(add_context_to_examples, batched=True)

    predictions = postprocess_qa_predictions(
        examples=predict_examples,
        features=predict_dataset,
        predictions=outputs_numpy,
        n_best_size=args.n_best_size,
        max_answer_length=args.max_answer_length
    )

    # Save results to CSV
    if accelerator.is_main_process:
        predict_id = list(predictions.keys())
        predict_result = list(predictions.values())

        df = pd.DataFrame({'id': predict_id, 'answer': predict_result})
        df.to_csv(args.output_file, index=False)
        logger.info(f"Predictions saved to {args.output_file}")


if __name__ == "__main__":
    main()