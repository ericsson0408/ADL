"""
Generate hard negatives using the current retriever for reranker training.
This ensures training/test distribution consistency.
"""
import json
import os
import faiss
import sqlite3
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--train_path", default="./data/train.txt")
parser.add_argument("--qrels_path", default="./data/qrels.txt")
parser.add_argument("--retriever_model_path", required=True)
parser.add_argument("--index_folder", required=True)
parser.add_argument("--index_file", default="passage_index.faiss")
parser.add_argument("--sqlite_file", default="passage_store.db")
parser.add_argument("--output_path", default="./data/train_with_hard_negs.jsonl")
parser.add_argument("--top_k", type=int, default=100, help="Retrieve top-K candidates")
args = parser.parse_args()

DEVICE = "cuda:0"
BATCH_SIZE = 64

print("Loading retriever...")
retriever = SentenceTransformer(args.retriever_model_path, device=DEVICE)

print("Loading FAISS index...")
index = faiss.read_index(os.path.join(args.index_folder, args.index_file))

print("Loading SQLite database...")
sqlite_path = f"{args.index_folder}/{args.sqlite_file}"
conn = sqlite3.connect(sqlite_path)
cur = conn.cursor()

print("Loading qrels...")
with open(args.qrels_path, "r") as f:
    qrels = json.load(f)
qid2gold = {}
for qid, pid_dict in qrels.items():
    gold_pids = {pid for pid, rel in pid_dict.items() if int(rel) > 0}
    qid2gold[qid] = gold_pids

print("Loading training data...")
train_data = []
with open(args.train_path, "r") as f:
    for line in f:
        if line.strip():
            train_data.append(json.loads(line))

print(f"Processing {len(train_data)} training queries...")

output_data = []

for i in tqdm(range(0, len(train_data), BATCH_SIZE)):
    batch = train_data[i:i+BATCH_SIZE]
    
    queries = [item["rewrite"] for item in batch]
    qids = [item["qid"] for item in batch]
    
    # Retrieve with current retriever
    prefix_queries = ["query: " + q for q in queries]
    q_embs = retriever.encode(
        prefix_queries,
        convert_to_numpy=True,
        normalize_embeddings=True,
        batch_size=BATCH_SIZE
    )
    
    D, I = index.search(q_embs, args.top_k)
    
    # Get passage texts
    need_rowids = set(int(rid) for row in I for rid in row.tolist())
    placeholders = ",".join(["?"] * len(need_rowids))
    sql = f"SELECT rowid, pid, text FROM passages WHERE rowid IN ({placeholders})"
    rows = cur.execute(sql, tuple(need_rowids)).fetchall()
    rowid2pt = {rid: (pid, text) for (rid, pid, text) in rows}
    
    # Create training examples
    for j, (qid, query, row_ids) in enumerate(zip(qids, queries, I)):
        gold_pids = qid2gold.get(qid, set())
        
        retrieved_passages = []
        labels = []
        
        for rid in row_ids:
            tup = rowid2pt.get(int(rid))
            if tup is None:
                continue
            pid, text = tup
            
            # Label: 1 if in gold set, 0 otherwise
            label = 1 if pid in gold_pids else 0
            
            retrieved_passages.append(text)
            labels.append(label)
        
        # Create output item
        output_item = {
            "qid": qid,
            "rewrite": query,
            "evidences": retrieved_passages,
            "retrieval_labels": labels,
            "gold_pids": list(gold_pids)
        }
        output_data.append(output_item)

conn.close()

print(f"\nWriting {len(output_data)} items to {args.output_path}...")
with open(args.output_path, "w", encoding="utf-8") as f:
    for item in output_data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print("Done!")
print(f"\nNow train your reranker with:")
print(f"  --train_path {args.output_path}")
