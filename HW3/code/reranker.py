import argparse
import json
import logging
import os
import random
from collections import defaultdict
from tqdm import tqdm
from torch.utils.data import DataLoader
from sentence_transformers import CrossEncoder, InputExample
# --- 【CORRECTED IMPORT】---
from sentence_transformers.cross_encoder.evaluation import (
    CEBinaryClassificationEvaluator, 
    CERerankingEvaluator
)
from sentence_transformers import evaluation
from sklearn.model_selection import train_test_split

# Set seed for reproducibility
SEED = 42
random.seed(SEED)

# ===========================
# Custom Loss Tracker for CrossEncoder
# ===========================
class LossTrackingCrossEncoder(CrossEncoder):
    """CrossEncoder with training loss tracking and IR metrics computation"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_losses = []
        self.all_step_losses = []  # Store loss at each eval step
        self.all_step_metrics = []  # Store IR metrics at each eval step
        self.step_numbers = []  # Store step numbers for plotting
        self.dev_queries = None
        
    def compute_ir_metrics(self):
        """Compute IR metrics (MRR, NDCG, MAP) on dev set"""
        if not self.dev_queries:
            return {}
        
        from sklearn.metrics import ndcg_score
        import numpy as np
        
        mrr_scores = []
        ndcg_scores = []
        map_scores = []
        accuracy_at_10 = []
        
        for query, passages in self.dev_queries.items():
            if not passages:
                continue
                
            # Score all passages for this query
            pairs = [(query, p[0]) for p in passages]
            scores = self.predict(pairs)
            labels = np.array([p[1] for p in passages])
            
            # Sort by scores (descending)
            sorted_indices = np.argsort(-scores)
            sorted_labels = labels[sorted_indices]
            
            # MRR: Mean Reciprocal Rank
            relevant_positions = np.where(sorted_labels > 0.5)[0]
            if len(relevant_positions) > 0:
                mrr_scores.append(1.0 / (relevant_positions[0] + 1))
                accuracy_at_10.append(1.0 if relevant_positions[0] < 10 else 0.0)
            else:
                mrr_scores.append(0.0)
                accuracy_at_10.append(0.0)
            
            # NDCG@10
            if len(sorted_labels) > 0:
                # Truncate to top 10
                top10_labels = sorted_labels[:10]
                # NDCG requires 2D arrays
                try:
                    ndcg = ndcg_score([labels[:10]], [scores[sorted_indices][:10]], k=10)
                    ndcg_scores.append(ndcg)
                except:
                    ndcg_scores.append(0.0)
            
            # MAP: Mean Average Precision
            relevant_indices = np.where(sorted_labels > 0.5)[0]
            if len(relevant_indices) > 0:
                precisions = []
                for i, rel_idx in enumerate(relevant_indices):
                    if rel_idx < 10:  # MAP@10
                        precision_at_k = (i + 1) / (rel_idx + 1)
                        precisions.append(precision_at_k)
                map_scores.append(np.mean(precisions) if precisions else 0.0)
            else:
                map_scores.append(0.0)
        
        metrics = {
            'mrr@10': np.mean(mrr_scores) if mrr_scores else 0.0,
            'ndcg@10': np.mean(ndcg_scores) if ndcg_scores else 0.0,
            'map@10': np.mean(map_scores) if map_scores else 0.0,
            'accuracy@10': np.mean(accuracy_at_10) if accuracy_at_10 else 0.0
        }
        
        return metrics
    
    def fit(self, *args, **kwargs):
        """Override fit to track losses and compute IR metrics at each evaluation"""
        # Store original callback if provided
        original_callback = kwargs.get('callback', None)
        
        # Create custom callback that tracks losses and computes IR metrics
        def loss_tracking_callback(score, epoch, steps):
            # Calculate and store current average loss
            if self.batch_losses:
                avg_loss = sum(self.batch_losses) / len(self.batch_losses)
                self.all_step_losses.append(avg_loss)
                self.step_numbers.append(steps)
                logging.info(f"\n[Epoch {epoch}, Step {steps}] Average Training Loss: {avg_loss:.4f}")
                self.batch_losses = []
            
            # Compute IR metrics
            if self.dev_queries:
                logging.info(f"[Epoch {epoch}, Step {steps}] Computing IR metrics...")
                ir_metrics = self.compute_ir_metrics()
                self.all_step_metrics.append(ir_metrics)
                logging.info(f"[Epoch {epoch}, Step {steps}] MRR@10: {ir_metrics.get('mrr@10', 0):.4f}, "
                           f"NDCG@10: {ir_metrics.get('ndcg@10', 0):.4f}, "
                           f"MAP@10: {ir_metrics.get('map@10', 0):.4f}")
            
            # Call original callback if it exists
            if original_callback:
                original_callback(score, epoch, steps)
        
        kwargs['callback'] = loss_tracking_callback
        return super().fit(*args, **kwargs)

def main():
    # --- 1. 參數解析 ---
    parser = argparse.ArgumentParser(description='Train Cross-Encoder Reranker')
    parser.add_argument("--train_path", default="train.txt", help="Path to training data JSONL file")
    parser.add_argument("--corpus_path", default="corpus.txt", help="Path to corpus JSONL file")
    parser.add_argument("--qrels_path", default="qrels.txt", help="Path to qrels JSON file")
    parser.add_argument("--model_name", default="cross-encoder/ms-marco-MiniLM-L-12-v2", help="Pretrained model name")
    parser.add_argument("--batch_size", type=int, default=32, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--eval_steps", type=int, default=500, help="Evaluation steps")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio")
    parser.add_argument("--dev_size", type=float, default=0.1, help="Development set size ratio")
    parser.add_argument("--max_length", type=int, default=384, help="Maximum sequence length")
    parser.add_argument("--save_path", default="reranker-best-model", help="Path to save best model")
    parser.add_argument("--max_negs", type=int, default=7, help="Maximum negatives per query to use")
    parser.add_argument("--grad_accum", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument("--use_hard_negatives", action="store_true", default=True, 
                        help="Use hard negatives from evidences")
    args = parser.parse_args()

    # --- 2. 初始化與載入資料 ---
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO)
    os.makedirs(args.save_path, exist_ok=True)

    logging.info("Loading corpus...")
    corpus = {}
    with open(args.corpus_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Reading Corpus"):
            data = json.loads(line)
            corpus[data['id']] = data['text']
    logging.info(f"Loaded {len(corpus)} passages from corpus.")

    logging.info("Loading qrels...")
    with open(args.qrels_path, 'r', encoding='utf-8') as f:
        qrels_data = json.load(f)
        qrels_map = {}
        for qid, pid_dict in qrels_data.items():
            correct_pid = list(pid_dict.keys())[0]
            qrels_map[qid] = correct_pid
    logging.info(f"Loaded {len(qrels_map)} query-passage relations from qrels.")

    # --- 3. 建構訓練與驗證資料 ---
    train_samples = []
    query_to_samples = defaultdict(list)  # Track samples per query
    
    logging.info("Reading training data and creating examples...")
    with open(args.train_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Processing Train Data"):
            data = json.loads(line)
            query = data['rewrite']
            qid = data['qid']

            # Get positive from qrels + corpus
            positive_added = False
            if qid in qrels_map:
                positive_pid = qrels_map[qid]
                if positive_pid in corpus:
                    positive_passage = corpus[positive_pid]
                    sample = InputExample(texts=[query, positive_passage], label=1.0)
                    train_samples.append(sample)
                    query_to_samples[qid].append(sample)
                    positive_added = True

            evidences = data.get('evidences', [])
            labels = data.get('retrieval_labels', [])
            
            # Add hard negatives from evidences (these are what the retriever returned)
            neg_count = 0
            for passage_text, label in zip(evidences, labels):
                if label == 0 and neg_count < args.max_negs:
                    sample = InputExample(texts=[query, passage_text], label=0.0)
                    train_samples.append(sample)
                    query_to_samples[qid].append(sample)
                    neg_count += 1
                elif label == 1 and not positive_added:
                    # Use positive from evidences if not found in corpus
                    sample = InputExample(texts=[query, passage_text], label=1.0)
                    train_samples.append(sample)
                    query_to_samples[qid].append(sample)
                    positive_added = True
    
    logging.info(f"Created a total of {len(train_samples)} examples from {len(query_to_samples)} queries.")
    
    # Calculate positive/negative ratio
    n_pos = sum(1 for s in train_samples if s.label == 1.0)
    n_neg = sum(1 for s in train_samples if s.label == 0.0)
    logging.info(f"Positive samples: {n_pos}, Negative samples: {n_neg}, Ratio: 1:{n_neg/max(n_pos,1):.2f}")

    # --- 4. 切分訓練集與驗證集 (by query to avoid leakage) ---
    logging.info(f"Splitting data into training and development sets by query...")
    
    # Split queries, not samples
    all_qids = list(query_to_samples.keys())
    train_qids, dev_qids = train_test_split(
        all_qids,
        test_size=args.dev_size,
        random_state=SEED,
        shuffle=True
    )
    
    # Assign samples based on query split
    train_samples_split = []
    dev_samples_split = []
    
    for qid in train_qids:
        train_samples_split.extend(query_to_samples[qid])
    
    for qid in dev_qids:
        dev_samples_split.extend(query_to_samples[qid])
    
    logging.info(f"Training samples: {len(train_samples_split)} from {len(train_qids)} queries")
    logging.info(f"Development samples: {len(dev_samples_split)} from {len(dev_qids)} queries")

    # --- 5. 初始化模型與 DataLoader ---
    logging.info(f"Loading cross-encoder model: {args.model_name}")
    model = LossTrackingCrossEncoder(args.model_name, num_labels=1, max_length=args.max_length)
    train_dataloader = DataLoader(train_samples_split, shuffle=True, batch_size=args.batch_size)

    # --- 6. 構建評估器 (使用 Binary Classification Evaluator 搭配自訂 IR metrics) ---
    logging.info("Setting up evaluation with Binary Classification + Manual IR Metrics...")
    
    # Use standard binary classification evaluator
    evaluator = CEBinaryClassificationEvaluator.from_input_examples(
        dev_samples_split,
        name='dev-split'
    )
    
    # Organize dev data by query for IR metrics computation
    dev_queries_dict = defaultdict(list)
    for sample in dev_samples_split:
        query_text = sample.texts[0]
        passage_text = sample.texts[1]
        label = sample.label
        dev_queries_dict[query_text].append((passage_text, label))
    
    # Store dev queries in model for IR metrics computation
    model.dev_queries = dev_queries_dict
    
    logging.info(f"Dev set: {len(dev_samples_split)} samples from {len(dev_queries_dict)} unique queries")

    # --- 7. 開始訓練 ---
    warmup_steps = int(len(train_dataloader) * args.epochs * args.warmup_ratio)
    logging.info(f"Warmup steps: {warmup_steps}")
    logging.info(f"Gradient accumulation steps: {args.grad_accum}")
    logging.info("Starting model training with evaluation...")

    try:
        model.fit(
            train_dataloader=train_dataloader,
            evaluator=evaluator,
            epochs=args.epochs,
            evaluation_steps=args.eval_steps,
            warmup_steps=warmup_steps,
            output_path=args.save_path,
            save_best_model=True,
            optimizer_params={'lr': args.lr},
            show_progress_bar=True,
            gradient_accumulation_steps=args.grad_accum,
        )
    except TypeError:
        # Fallback for older versions
        logging.warning("gradient_accumulation_steps not supported, training without it...")
        model.fit(
            train_dataloader=train_dataloader,
            evaluator=evaluator,
            epochs=args.epochs,
            evaluation_steps=args.eval_steps,
            warmup_steps=warmup_steps,
            output_path=args.save_path,
            save_best_model=True,
            optimizer_params={'lr': args.lr},
            show_progress_bar=True,
        )

    logging.info(f"Training complete. The best model has been saved to {args.save_path}")
    
    # --- 8. 保存訓練 metrics 供繪圖使用 ---
    metrics_file = os.path.join(args.save_path, "training_metrics.json")
    
    # Extract IR metrics from model (at each eval step)
    mrr_list = []
    ndcg_list = []
    map_list = []
    accuracy_list = []
    
    if hasattr(model, 'all_step_metrics') and model.all_step_metrics:
        for metrics_dict in model.all_step_metrics:
            mrr_list.append(metrics_dict.get('mrr@10', 0.0))
            ndcg_list.append(metrics_dict.get('ndcg@10', 0.0))
            map_list.append(metrics_dict.get('map@10', 0.0))
            accuracy_list.append(metrics_dict.get('accuracy@10', 0.0))
    
    metrics_data = {
        "steps": model.step_numbers if hasattr(model, 'step_numbers') else [],
        "train_loss": model.all_step_losses if hasattr(model, 'all_step_losses') else [],
        "mrr": mrr_list,
        "ndcg": ndcg_list,
        "map": map_list,
        "accuracy": accuracy_list
    }
    
    # Save to JSON
    try:
        with open(metrics_file, 'w') as f:
            json.dump(metrics_data, f, indent=2)
        logging.info(f"\n{'='*60}")
        logging.info(f"Training metrics saved to: {metrics_file}")
        logging.info(f"  - Evaluation points: {len(metrics_data['steps'])}")
        logging.info(f"  - Steps: {metrics_data['steps']}")
        logging.info(f"  - Training losses: {metrics_data['train_loss']}")
        if metrics_data['mrr']:
            logging.info(f"  - MRR@10: {metrics_data['mrr']}")
            logging.info(f"  - NDCG@10: {metrics_data['ndcg']}")
            logging.info(f"  - MAP@10: {metrics_data['map']}")
            logging.info(f"  - Accuracy@10: {metrics_data['accuracy']}")
        logging.info(f"{'='*60}")
    except Exception as e:
        logging.warning(f"Could not save training metrics: {e}")

if __name__ == "__main__":
    main()
