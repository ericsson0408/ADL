import numpy as np
import json, faiss, torch
from sentence_transformers import SentenceTransformer, CrossEncoder, util
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
from tqdm import tqdm
from huggingface_hub import login
from dotenv import load_dotenv
import os
import sqlite3
import re
from utils import get_inference_user_prompt, get_inference_system_prompt, parse_generated_answer
import gc

# --- 【新增】 匯入 Stable Baselines 3 ---
from stable_baselines3 import PPO 

# --- (登入部分... 與原檔案相同) ---
load_dotenv()
hf_token = os.getenv("hf_token")
login(token=hf_token)

argparser = argparse.ArgumentParser()
argparser.add_argument("--data_folder", type=str, default="./data")
argparser.add_argument("--passage_file", type=str, default="corpus.txt")
argparser.add_argument("--index_folder", type=str, default="./vector_database")
argparser.add_argument("--index_file", type=str, default="passage_index.faiss")
argparser.add_argument("--sqlite_file", type=str, default="passage_store.db")
argparser.add_argument("--test_data_path", type=str, default="./data/test_open.txt")
argparser.add_argument("--qrels_path", type=str, default="./data/qrels.txt")
argparser.add_argument("--retriever_model_path", type=str, default="")
argparser.add_argument("--reranker_model_path", type=str, default="")
argparser.add_argument("--generator_model", type=str, default="Qwen/Qwen3-1.7B")
argparser.add_argument("--result_file_name", type=str, default="result_rl.json") # 改個名字
# --- 【新增】 RL Agent 路徑參數 ---
argparser.add_argument("--rl_agent_path", type=str, default="rag_rl_agent.zip")
args = argparser.parse_args()

data_folder = args.data_folder
passage_file = args.passage_file
index_folder = args.index_folder
index_file = args.index_file
sqlite_file = args.sqlite_file
test_data_path = args.test_data_path
retriever_model_path = args.retriever_model_path
reranker_model_path = args.reranker_model_path
qrels_path = args.qrels_path
result_file = args.result_file_name

###############################################################################
# 0. parameters
TOP_K = 10
# --- 【修改】 移除固定的 TOP_M ---
# TOP_M = 3 
GEN_MAXLEN = 1280
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_Q = 64 
BATCH_GEN = 32 
TEST_DATA_SIZE = -1  

###############################################################################
# 1. load db and index
# --- (此區塊... 與原檔案相同) ---
sqlite_path = f"{index_folder}/{sqlite_file}"
conn = sqlite3.connect(sqlite_path)
cur = conn.cursor()
retriever = SentenceTransformer(retriever_model_path, device=DEVICE)
vram_allocated = torch.cuda.memory_stats()["allocated_bytes.all.current"]
print(f"Retriever VRAM: {vram_allocated/1e9:.2f} GB")
index = faiss.read_index(os.path.join(index_folder, index_file))

###############################################################################
# 2. load dataset
# --- (此區塊... 與原檔案相同) ---
def load_qrels_gold_pids(qrels_path):
    with open(qrels_path, "r", encoding="utf-8") as f:
        qrels = json.load(f)
    qid2gold = {}
    for qid, pid2lab in qrels.items():
        gold = {pid for pid, lab in pid2lab.items() if str(lab) != "0"}
        qid2gold[qid] = gold
    return qid2gold

tests = []  
qid2gold = load_qrels_gold_pids(qrels_path)
with open(test_data_path, "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        obj = json.loads(line)
        qid = obj.get("qid")
        query = obj.get("rewrite")
        gold_answer = (obj.get("answer")).get("text", "")
        gold_pids = qid2gold.get(qid, set())
        tests.append({"qid": qid, "query": query, "gold_answer": gold_answer, "gold_pids": gold_pids})
tests = tests[:TEST_DATA_SIZE]

# =========================
# 3. eval metrices（Recall / MRR）
# --- (此區塊... 與原檔案相同) ---
def recall_at_k(retrieved_pids, gold_pids, k):
    topk = retrieved_pids[:k]
    return 1.0 if any(pid in gold_pids for pid in topk) else 0.0

def mrr_at_k(reranked_pids, gold_pids, k):
    for rank, pid in enumerate(reranked_pids[:k]):
        if pid in gold_pids:
            score = 1.0 / (rank + 1)
            return score
    return 0.0
# =========================
# 4. inference
# =========================

reranker = CrossEncoder(reranker_model_path, device=DEVICE)
print(f"Reranker VRAM: {(torch.cuda.memory_stats()['allocated_bytes.all.current'] - vram_allocated)/1e9:.2f} GB")
vram_allocated = torch.cuda.memory_stats()["allocated_bytes.all.current"]

model_id = args.generator_model
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, dtype="auto", device_map="auto")
print(f"LLM VRAM: {(torch.cuda.memory_stats()['allocated_bytes.all.current'] - vram_allocated)/1e9:.2f} GB")
vram_allocated = torch.cuda.memory_stats()["allocated_bytes.all.current"]

# --- 【新增】 載入訓練好的 RL Agent ---
print(f"Loading RL Agent from {args.rl_agent_path}")
rl_agent = PPO.load(args.rl_agent_path, device=DEVICE)

R_at_K_sum = 0.0
MRR_sum = 0.0
N = 0

output_records = []
for b_start in tqdm(range(0, len(tests), BATCH_Q)):
    batch = tests[b_start:b_start+BATCH_Q]
    qids        = [ex["qid"] for ex in batch]
    queries     = [ex["query"] for ex in batch]
    gold_sets   = [ex["gold_pids"] for ex in batch]
    gold_ans    = [ex["gold_answer"] for ex in batch]

    # 1) retrieve and search from FAISS
    prefix_queries = ["query: " + q for q in queries] 
    q_embs = retriever.encode(
        prefix_queries, convert_to_numpy=True, normalize_embeddings=True,
        batch_size=BATCH_Q
    )
    D, I = index.search(q_embs, TOP_K) 

    

    # 2) retrieve all passages needed in this batch
    # --- (此區塊... 與原檔案相同) ---
    need_rowids = set(int(rid) for row in I for rid in row.tolist())
    placeholders = ",".join(["?"] * len(need_rowids)) or "NULL"
    sql = f"SELECT rowid, pid, text FROM passages WHERE rowid IN ({placeholders})"
    rows = cur.execute(sql, tuple(need_rowids)).fetchall()
    rowid2pt = {rid: (pid, text) for (rid, pid, text) in rows}

    # 3) create（cand_ids, cand_texts）of each query and calculate recall@K
    # --- (此區塊... 與原檔案相同) ---
    batch_cand_ids, batch_cand_texts = [], []
    for b, row in enumerate(I):
        rid_list = row.tolist()
        cand_ids, cand_texts = [], []
        for rid in rid_list:
            tup = rowid2pt.get(int(rid))
            if tup is None: continue
            pid, text = tup
            cand_ids.append(pid)
            cand_texts.append(text)
        batch_cand_ids.append(cand_ids)
        batch_cand_texts.append(cand_texts)
        R_at_K_sum += recall_at_k(cand_ids, gold_sets[b], TOP_K)

    # 4) create (query, passage) pairs into flat list for reranker.predict
    # --- (此區塊... 與原檔案相同) ---
    flat_pairs = []
    idx_slices = []   
    cursor = 0
    for q, ctexts in zip(queries, batch_cand_texts):
        n = len(ctexts)
        if n == 0:
            idx_slices.append((cursor, cursor))
            continue
        flat_pairs.extend(zip([q] * n, ctexts))
        idx_slices.append((cursor, cursor + n))
        cursor += n

    if len(flat_pairs) == 0:
        MRR_sum      += 0.0 * len(batch)
        N            += len(batch)
        continue

    # 5) reranker score all pairs in flat list
    # --- (此區塊... 與原檔案相同) ---
    flat_scores = reranker.predict(flat_pairs)

    # 6) 建立 RL State, 預測 M 值, 並建立 Generation Prompt
    
    batch_agent_states = []
    all_reranked_data = [] # 用來暫存排序後的 (score, pid, text)
    rerank_info_list_for_json = [] # 用來暫存給 JSON 輸出的 rerank info

    # --- 6.1) 第一個循環：計算 State 並計算 MRR ---
    for b, (q, (low, high)) in enumerate(zip(queries, idx_slices)):
        if low == high: 
            MRR_sum += 0.0
            N += 1
            # 必須 append 一個 "dummy" state 和 "None" data 來保持索引對齊
            batch_agent_states.append(np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)) 
            all_reranked_data.append(None)
            rerank_info_list_for_json.append(None)
            continue

        scores = flat_scores[low:high]
        cand_ids  = batch_cand_ids[b]
        cand_text = batch_cand_texts[b]
        
        # 排序 (我們需要 (score, pid, text) 格式)
        reranked = sorted(zip(scores, cand_ids, cand_text), key=lambda x: x[0], reverse=True)
        all_reranked_data.append(reranked) # 儲存排序結果
        
        # 計算 MRR (使用所有 Top-K 文件)
        reranked_pids = [pid for _, pid, _ in reranked]
        MRR_sum += mrr_at_k(reranked_pids, gold_sets[b], TOP_K)
        
        # --- 建立 4-D State (複製 RAGEnv._get_obs 的邏輯) ---
        reranked_scores = [s for s, _, _ in reranked]
        
        # 處理邊界情況 (必須與 train_rl.py 中的 _get_obs 完全一致)
        if not reranked_scores:
            reranked_scores = [0.0, 0.0]
        elif len(reranked_scores) == 1:
            reranked_scores.append(0.0)

        feature_1 = reranked_scores[0]
        feature_2 = reranked_scores[0] - reranked_scores[1]
        feature_3 = np.mean(reranked_scores) # Top-K (K=10) 的 mean
        feature_4 = np.std(reranked_scores)  # Top-K (K=10) 的 std
        
        state = np.array([feature_1, feature_2, feature_3, feature_4], dtype=np.float32)
        batch_agent_states.append(state)
        # --- State 建立完畢 ---

        # 儲存 JSON 用的 rerank info
        rerank_info_list_for_json.append([
            {"pid": pid, "text": text, "score": float(score)}
            for score, pid, text in reranked
        ])

    # --- 6.2) 批次預測動態 M 值 ---
    if not batch_agent_states: # 如果整個批次都是空的
        N += len(batch) # 補上計數
        continue

    np_states = np.stack(batch_agent_states) # Shape (BATCH_Q, 4)
    
    # ！！！ 這才是正確的 predict 呼叫 ！！！
    actions, _ = rl_agent.predict(np_states, deterministic=True)
    
    # actions 是 (BATCH_Q,) array, 內容是 0-4 (因為 ActionSpace 是 Discrete(5))
    # M 值是 action + 1 (範圍 1-5)
    dynamic_top_m_list = actions + 1

    # --- 6.3) 第二個循環：使用 M 值建立 Prompt ---
    messages_list = []
    for b in range(len(batch)):
        reranked = all_reranked_data[b] # 取得先前儲存的排序結果
        
        if reranked is None: # 這是之前跳過的空查詢
            messages_list.append(None)
            continue
            
        # 取得這個 query 該用的 M 值
        dynamic_m = dynamic_top_m_list[b] 
        
        # 根據 dynamic_m (1-5) 選取文件
        context_list = [text for score, pid, text in reranked[:min(dynamic_m, len(reranked))]]
        
        messages = [
            {
                "role": "system",
                "content": get_inference_system_prompt()
            },
            {
                "role": "user",
                "content": get_inference_user_prompt(queries[b], context_list)
            }
        ]
        messages_list.append(messages)
    
    # 7) generation for all queries in this batch
    # ... (此區塊保持不變, 但 'rerank_info_list' 變數名稱需統一) ...
    pending = [(idx, passage) for idx, passage in enumerate(messages_list) if passage is not None]
    for g_start in range(0, len(pending), BATCH_GEN):
        chunk = pending[g_start:g_start+BATCH_GEN]
        idxs, msgs_batch = zip(*chunk)
        tokenizer.padding_side = "left"
        rendered_prompts = [
            tokenizer.apply_chat_template(
                m, add_generation_prompt=True, tokenize=False, enable_thinking=False
            )
            for m in msgs_batch
        ]

        inputs = tokenizer(
            rendered_prompts, padding=True, return_tensors="pt"
        ).to(model.device)
        with torch.no_grad():
            outs = model.generate(**inputs, max_new_tokens=GEN_MAXLEN)
        decoded = tokenizer.batch_decode(outs, skip_special_tokens=True)
        for j, ans in zip(idxs, decoded):
            pred_ans = parse_generated_answer(ans.strip())
            
            # 【注意】: 這裡 N += 1 應該被移除，因為我們在 6.1) 已經加過了
            # N += 1 (移除這行)
            
            output_records.append({
                "qid": qids[j],
                "query": queries[j],
                "retrieved": rerank_info_list_for_json[j], # <-- 使用新的變數名稱
                "generated": pred_ans,
                "gold_answer": gold_ans[j],
                "faiss_distances": D[j].tolist(),
                "faiss_rowids": I[j].tolist(),
                "gold_pids": list(gold_sets[j])
            })
        
# --- (釋放記憶體區塊... 與原檔案相同) ---
del model
del retriever
del reranker
del rl_agent # <-- 【新增】 釋放 RL Agent
gc.collect()
torch.cuda.empty_cache()

# --- (計算分數區塊... 與原檔案相同) ---
res = [record["generated"] for record in output_records]
ans = [record["gold_answer"] for record in output_records]

sentence_scorer = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=DEVICE)
emb_res  = sentence_scorer.encode(res,  convert_to_tensor=True, normalize_embeddings=True)
emb_gold = sentence_scorer.encode(ans, convert_to_tensor=True, normalize_embeddings=True)
scores = util.cos_sim(emb_res, emb_gold)
diag_scores = scores.diag().tolist()
bi_encoder_similarity = np.mean(diag_scores)
# =========================
# 5. results
# =========================
print(f"Queries evaluated: {N}")
print(f"Recall@{TOP_K}: {R_at_K_sum / max(N,1):.4f}")
print(f"MRR@{TOP_K} (after rerank): {MRR_sum / max(N,1):.4f}")
print(f"Bi-Encoder CosSim: {bi_encoder_similarity:.4f}")

final = {"data_size": N,
         f"recall@{TOP_K}": R_at_K_sum / max(N,1),
         f"mrr@{TOP_K}": MRR_sum / max(N,1),
        "Bi-Encoder_CosSim": bi_encoder_similarity,
         "records": output_records}

os.makedirs("results", exist_ok=True)
result_file_path = os.path.join("results", result_file)
with open(result_file_path, "w", encoding="utf-8") as f:
    json.dump(final, f, indent=2, ensure_ascii=False)