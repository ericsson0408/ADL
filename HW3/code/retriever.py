import json
import os
import random
from datetime import datetime
from typing import List, Dict, Tuple

from sentence_transformers import SentenceTransformer, InputExample, losses, evaluation
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import torch

# Reduce noisy tokenizer warnings
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# -----------------------------
# Data utilities
# -----------------------------
def _normalize_query(q: str) -> str:
    q = (q or "").strip()
    return q if q.startswith("query: ") else f"query: {q}"

def _normalize_passage(p: str) -> str:
    p = (p or "").strip()
    return p if p.startswith("passage: ") else f"passage: {p}"

def _read_label_array(d: dict) -> List[int]:
    # Be forgiving about the key name in the JSONL input
    for k in ["retrieval_labels", "retrieva_labels", "labels", "gold_labels", "y"]:
        if k in d and isinstance(d[k], list):
            return d[k]
    raise KeyError("No label array found. Expected one of: retrieval_labels / labels / gold_labels / y")

def load_pairs_for_mnrl(train_path: str, max_pos_per_query: int = 4) -> List[InputExample]:
    """
    Load training pairs (query, positive_passage) from a JSONL file.
    Each line: {
        "rewrite": <query str>,
        "evidences": [<passage1>, <passage2>, ...],
        "retrieval_labels": [0/1, 0/1, ...]   # 1 = positive
    }
    """
    pairs: List[InputExample] = []
    n_lines = 0
    n_queries = 0
    n_pos = 0

    with open(train_path, "r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            n_lines += 1
            try:
                row = json.loads(raw)
            except Exception:
                # Skip broken lines quietly
                continue

            q = _normalize_query(row.get("rewrite", ""))
            evidences = row.get("evidences", [])
            try:
                labels = _read_label_array(row)
            except KeyError:
                # If there are no labels, consider the first passage as positive (fallback)
                labels = [1] + [0] * (len(evidences) - 1)

            if not evidences:
                continue

            # Collect positives (capped to avoid huge explosion on multi-label data)
            pos_passages = [_normalize_passage(p) for p, y in zip(evidences, labels) if y == 1][:max_pos_per_query]
            if not pos_passages:
                # If no positive labels, skip this query
                continue

            n_queries += 1
            for p in pos_passages:
                pairs.append(InputExample(texts=[q, p]))
                n_pos += 1

    if n_queries == 0:
        print("No usable training samples found.")
    else:
        print(f"Loaded {n_pos} (query,positive) pairs from {n_lines} lines ({n_queries} queries).")

    return pairs


def load_evaluation_data(
    train_path: str, 
    qrels_path: str = None, 
    corpus_path: str = None,
    dev_ratio: float = 0.1
) -> Tuple[Dict[str, str], Dict[str, set], Dict[str, str]]:
    """
    Load evaluation data for Information Retrieval evaluation.
    Returns: (queries, relevant_docs, corpus)
    """
    queries = {}
    relevant_docs = {}
    corpus = {}
    
    # Load corpus if provided
    if corpus_path and os.path.exists(corpus_path):
        print(f"Loading corpus from {corpus_path}...")
        with open(corpus_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    doc = json.loads(line)
                    doc_id = doc.get("id", "")
                    text = doc.get("text", "")
                    title = doc.get("title", "")
                    if doc_id and text:
                        corpus[doc_id] = _normalize_passage(f"{title} {text}".strip())
                except:
                    continue
        print(f"Loaded {len(corpus)} documents from corpus.")
    
    # Load qrels if provided
    if qrels_path and os.path.exists(qrels_path):
        print(f"Loading qrels from {qrels_path}...")
        with open(qrels_path, "r", encoding="utf-8") as f:
            qrels_data = json.load(f)
            for qid, pid_dict in qrels_data.items():
                if qid not in relevant_docs:
                    relevant_docs[qid] = set()
                for pid, rel in pid_dict.items():
                    if rel > 0:
                        relevant_docs[qid].add(pid)
        print(f"Loaded {len(relevant_docs)} query-doc relations from qrels.")
    
    # Load queries from train file
    print(f"Loading queries from {train_path}...")
    all_queries = []
    with open(train_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                qid = row.get("qid", "")
                query_text = _normalize_query(row.get("rewrite", ""))
                if qid and query_text:
                    all_queries.append((qid, query_text))
            except:
                continue
    
    # Split for dev set
    if all_queries:
        random.seed(42)
        random.shuffle(all_queries)
        n_dev = max(1, int(len(all_queries) * dev_ratio))
        dev_queries = all_queries[:n_dev]
        
        for qid, query_text in dev_queries:
            queries[qid] = query_text
        
        print(f"Selected {len(queries)} queries for evaluation.")
    
    return queries, relevant_docs, corpus


# -----------------------------
# Training
# -----------------------------
def train_retriever(
    train_file: str = "train.txt",
    model_name: str = "intfloat/multilingual-e5-small",
    batch_size: int = 64,
    num_epochs: int = 3,
    lr: float = 2e-5,
    max_seq_len: int = 512,
    warmup_ratio: float = 0.1,
    output_dir: str = None,
    max_pos_per_query: int = 4,
    qrels_path: str = None,
    corpus_path: str = None,
    eval_steps: int = 500,
    use_evaluation: bool = True,
) -> str:
    print(f"Loading base model: {model_name}")
    model = SentenceTransformer(model_name)
    model.max_seq_length = max_seq_len

    print("Preparing data...")
    all_pairs = load_pairs_for_mnrl(train_file, max_pos_per_query=max_pos_per_query)
    if not all_pairs:
        raise SystemExit("No training data. Please check train.txt format.")
    
    # Split into train and dev for evaluation
    train_pairs = all_pairs
    dev_evaluator = None
    
    if use_evaluation and qrels_path and corpus_path:
        print("\nPreparing evaluation data...")
        # Split training pairs for validation
        train_pairs, dev_pairs = train_test_split(
            all_pairs, 
            test_size=0.1, 
            random_state=42,
            shuffle=True
        )
        
        # Load evaluation data
        queries, relevant_docs, corpus = load_evaluation_data(
            train_file, qrels_path, corpus_path, dev_ratio=0.1
        )
        
        if queries and corpus and relevant_docs:
            # Create IR evaluator
            dev_evaluator = evaluation.InformationRetrievalEvaluator(
                queries=queries,
                corpus=corpus,
                relevant_docs=relevant_docs,
                name="dev-ir-eval",
                show_progress_bar=True,
                batch_size=batch_size,
                mrr_at_k=[10],
                ndcg_at_k=[10],
                accuracy_at_k=[1, 3, 5, 10],
                precision_recall_at_k=[1, 3, 5, 10],
                map_at_k=[10],
            )
            print(f"Evaluation set ready: {len(queries)} queries, {len(corpus)} docs")
        else:
            print("Evaluation data incomplete, skipping evaluation.")

    train_loader = DataLoader(train_pairs, shuffle=True, batch_size=batch_size)
    # InfoNCE with in-batch negatives (= MultipleNegativesRankingLoss)
    train_loss = losses.MultipleNegativesRankingLoss(model)

    total_steps = len(train_loader) * num_epochs
    warmup_steps = max(1, int(total_steps * warmup_ratio))

    if output_dir is None:
        output_dir = f"finetuned-e5-mnr-{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    print("\n----- Training configuration -----")
    print(f"Training pairs: {len(train_pairs)} | Batch size: {batch_size}")
    print(f"Epochs: {num_epochs} | Total steps: {total_steps} | Warmup: {warmup_steps}")
    print(f"Max length: {max_seq_len} | LR: {lr}")
    print(f"Evaluation: {'Enabled' if dev_evaluator else 'Disabled'}")
    print(f"Output: {output_dir}")
    print("----------------------------------\n")

    # We'll use the standard model.fit() and extract loss from the training process
    # by wrapping the loss function
    class LossTracker(losses.MultipleNegativesRankingLoss):
        def __init__(self, model):
            super().__init__(model)
            self.epoch_losses = []
            self.current_losses = []
            
        def forward(self, sentence_features, labels):
            loss = super().forward(sentence_features, labels)
            self.current_losses.append(loss.item())
            return loss
        
        def on_epoch_end(self):
            if self.current_losses:
                avg_loss = sum(self.current_losses) / len(self.current_losses)
                self.epoch_losses.append(avg_loss)
                print(f"\n[Epoch {len(self.epoch_losses)}] Average Training Loss: {avg_loss:.4f}")
                self.current_losses = []
    
    # Use the tracked loss
    tracked_loss = LossTracker(model)
    
    # Custom callback to track epochs
    class EpochCallback:
        def __init__(self, loss_tracker):
            self.loss_tracker = loss_tracker
            
        def __call__(self, score, epoch, steps):
            self.loss_tracker.on_epoch_end()
    
    epoch_callback = EpochCallback(tracked_loss)
    
    model.fit(
        train_objectives=[(train_loader, tracked_loss)],
        evaluator=dev_evaluator,
        epochs=num_epochs,
        evaluation_steps=0,  # Only evaluate at epoch end
        optimizer_params={"lr": lr},
        warmup_steps=warmup_steps,
        output_path=output_dir,
        save_best_model=True,
        show_progress_bar=True,
        checkpoint_save_steps=5000,
        checkpoint_path=f"{output_dir}/ckpts",
        callback=epoch_callback,
    )

    # Save training metrics for plotting
    metrics_file = os.path.join(output_dir, "training_metrics.json")
    metrics_data = {
        "epochs": list(range(1, num_epochs + 1)),
        "train_loss": tracked_loss.epoch_losses,
        "mrr": [],
        "recall": [],
        "ndcg": [],
        "map": []
    }
    
    # Add evaluation metrics if available
    if dev_evaluator and hasattr(dev_evaluator, 'csv_file') and os.path.exists(dev_evaluator.csv_file):
        import csv
        try:
            with open(dev_evaluator.csv_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    epoch = int(row.get('epoch', 0))
                    if epoch > 0:
                        metrics_data["mrr"].append(float(row.get('mrr@10', 0)))
                        metrics_data["recall"].append(float(row.get('recall@10', 0)))
                        metrics_data["ndcg"].append(float(row.get('ndcg@10', 0)))
                        metrics_data["map"].append(float(row.get('map@10', 0)))
        except Exception as e:
            print(f"Warning: Could not read evaluation CSV: {e}")
    
    # Save to JSON
    try:
        with open(metrics_file, 'w') as f:
            json.dump(metrics_data, f, indent=2)
        print(f"\nTraining metrics saved to: {metrics_file}")
        print(f"  - Epochs: {len(metrics_data['epochs'])}")
        print(f"  - Training losses: {metrics_data['train_loss']}")
        if metrics_data['mrr']:
            print(f"  - MRR@10: {metrics_data['mrr']}")
    except Exception as e:
        print(f"Warning: Could not save training metrics: {e}")

    print(f"\nTraining finished. Model saved to: {output_dir}")
    return output_dir


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Retriever with MultipleNegativesRankingLoss')
    parser.add_argument("--train_path", "--train_file", dest="train_file", default="train.txt", 
                        help="Path to training data JSONL file")
    parser.add_argument("--model_name", default="intfloat/multilingual-e5-small", 
                        help="Pretrained SentenceTransformer model name")
    parser.add_argument("--batch_size", type=int, default=64, 
                        help="Training batch size")
    parser.add_argument("--epochs", type=int, default=3, 
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=2e-5, 
                        help="Learning rate")
    parser.add_argument("--max_seq_len", "--max_length", dest="max_seq_len", type=int, default=512, 
                        help="Maximum sequence length")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, 
                        help="Warmup ratio")
    parser.add_argument("--output_dir", "--save_path", dest="output_dir", default=None, 
                        help="Output directory for fine-tuned model")
    parser.add_argument("--max_pos_per_query", type=int, default=4, 
                        help="Maximum positive passages per query")
    parser.add_argument("--qrels_path", default=None,
                        help="Path to qrels file for evaluation")
    parser.add_argument("--corpus_path", default=None,
                        help="Path to corpus file for evaluation")
    parser.add_argument("--eval_steps", type=int, default=500,
                        help="Evaluation steps during training")
    parser.add_argument("--use_evaluation", action="store_true", default=False,
                        help="Enable evaluation during training")
    args = parser.parse_args()

    train_retriever(
        train_file=args.train_file,
        model_name=args.model_name,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        lr=args.lr,
        max_seq_len=args.max_seq_len,
        warmup_ratio=args.warmup_ratio,
        output_dir=args.output_dir,
        max_pos_per_query=args.max_pos_per_query,
        qrels_path=args.qrels_path,
        corpus_path=args.corpus_path,
        eval_steps=args.eval_steps,
        use_evaluation=args.use_evaluation,
    )
