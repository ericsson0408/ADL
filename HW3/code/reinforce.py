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
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# --- 登入與環境變數 (與 inference_batch.py 相同) ---
load_dotenv()
hf_token = os.getenv("hf_token")
login(token=hf_token)

# --- 1. RAG 環境定義 (Gymnasium Env) ---

class RAGEnv(gym.Env):
    """
    客製化的 RAG 強化學習環境
    
    State: 4-d Reranker 分數特徵
           (Top-1, Top-1/Top-2 Gap, Mean, Std)
    Action: 離散動作 0-4，對應到選擇 Top M=1 到 M=5 篇文件
    Reward: 生成答案與黃金答案的 Sentence Similarity
    """
    
    def __init__(self, precomputed_data, llm_model, llm_tokenizer, sentence_scorer, device="cuda"):
        super(RAGEnv, self).__init__()
        
        self.precomputed_data = precomputed_data
        self.llm_model = llm_model
        self.llm_tokenizer = llm_tokenizer
        self.sentence_scorer = sentence_scorer
        self.device = device
        
        self.current_idx = 0
        self.current_item = None
        
        # 動作空間: 5 個離散動作 (0=Top1, 1=Top2, ..., 4=Top5)
        self.action_space = spaces.Discrete(5) # <--- MODIFIED (從 10 改為 5)
        
        # 狀態空間: 4 維的 Reranker 分數特徵
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32) # <--- MODIFIED (從 384 改為 4)

    def _get_obs(self):
        # <--- MODIFIED (整個函式重寫) ---
        
        # 1. 從預處理資料中取得 Reranker 分數列表
        # precomputed_data 中的 'reranked_passages' 是一個 (text, score, pid) 的列表
        reranked_passages = self.current_item['reranked_passages']
        
        # 提取分數 (pre-processing 已確保是 Top-K=10 的分數)
        scores = [p[1] for p in reranked_passages]

        # 2. 處理邊界情況 (Edge Cases)
        # 確保至少有 2 個分數可用來計算 "gap"
        if not scores: # 如果列表為空
            scores = [0.0, 0.0]
        elif len(scores) == 1: # 如果只有 1 個文件
            scores.append(0.0) # 補上 0.0 以計算差距

        # 3. 計算 4-D State 特徵
        feature_1 = scores[0]               # 特徵1: Top-1 分數
        feature_2 = scores[0] - scores[1]   # 特徵2: Top-1 和 Top-2 的分數差距
        feature_3 = np.mean(scores)         # 特徵3: Top-K (K=10) 分數的平均值
        feature_4 = np.std(scores)          # 特徵4: Top-K (K=10) 分數的標準差

        # 4. 回傳狀態向量
        return np.array([feature_1, feature_2, feature_3, feature_4], dtype=np.float32)

    def _get_info(self):
        return {"qid": self.current_item['qid']}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # 隨機選擇一個訓練資料點作為新 episode
        self.current_idx = np.random.randint(0, len(self.precomputed_data))
        self.current_item = self.precomputed_data[self.current_idx]
        
        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def step(self, action):
        # action 是 0-4 的整數, M 是 1-5
        top_m = int(action) + 1 # <--- 這行邏輯保持不變，因為 action (0-4) + 1 剛好對應 M (1-5)
        
        # 1. 根據 action (M值) 選取文件
        query = self.current_item['query']
        reranked_passages = self.current_item['reranked_passages']
        # 這裡會從預存的 Top-10 列表中，選取 M=1 到 M=5 篇
        context_list = [text for text, score, pid in reranked_passages[:top_m]]

        # 2. 準備 Prompt 並呼叫 LLM
        messages = [
            {"role": "system", "content": get_inference_system_prompt()},
            {"role": "user", "content": get_inference_user_prompt(query, context_list)}
        ]
        self.llm_tokenizer.padding_side = "left"
        rendered_prompt = self.llm_tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False, enable_thinking=False
        )
        inputs = self.llm_tokenizer(rendered_prompt, padding=True, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outs = self.llm_model.generate(**inputs, max_new_tokens=1280) # GEN_MAXLEN
        
        decoded = self.llm_tokenizer.batch_decode(outs, skip_special_tokens=True)
        pred_ans = parse_generated_answer(decoded[0].strip())
        
        # 3. 計算 Reward (Sentence Similarity)
        gold_ans = self.current_item['gold_answer']
        
        if not pred_ans or not gold_ans: # 處理空字串
            reward = 0.0
        else:
            emb_res = self.sentence_scorer.encode(pred_ans, convert_to_tensor=True, normalize_embeddings=True)
            emb_gold = self.sentence_scorer.encode(gold_ans, convert_to_tensor=True, normalize_embeddings=True)
            score = util.cos_sim(emb_res, emb_gold)
            reward = float(score.diag().item())

        # 4. 回傳
        terminated = True  # 每個 episode 只包含一個 query-answer
        truncated = False
        info = self._get_info()
        observation = self._get_obs() # 下一個 state (雖然這局結束了)

        return observation, reward, terminated, truncated, info

# --- 2. 預處理函式 ---
# (此函式 'preprocess_training_data' 保持不變，
#  它仍然需要檢索 Top-K=10，以便 _get_obs 可以計算 mean 和 std)

def preprocess_training_data(retriever, reranker, index, conn, train_data, qid2gold, corpus):
    """
    預處理所有訓練資料，執行檢索和重排序，並儲存結果。
    這一步非常耗時，但是是必要的，以加速 RL 訓練。
    """
    print("Starting preprocessing of training data...")
    precomputed_data = []
    cur = conn.cursor()
    
    # 參數
    TOP_K = 10  # <--- 保持 K=10, 因為 4-D State 需要 Top-10 的 mean 和 std
    BATCH_Q = 256

    for b_start in tqdm(range(0, len(train_data), BATCH_Q), desc="Preprocessing Batches"):
        batch = train_data[b_start:b_start+BATCH_Q]
        queries = [ex["query"] for ex in batch]
        qids = [ex["qid"] for ex in batch]
        gold_ans = [ex["gold_answer"] for ex in batch]
        
        # 1) 檢索 (Retriever)
        prefix_queries = ["query: " + q for q in queries]
        q_embs = retriever.encode(
            prefix_queries, convert_to_numpy=True, normalize_embeddings=True, batch_size=BATCH_Q
        )
        D, I = index.search(q_embs, TOP_K)

        # 2) 取得文件
        need_rowids = set(int(rid) for row in I for rid in row.tolist())
        placeholders = ",".join(["?"] * len(need_rowids)) or "NULL"
        sql = f"SELECT rowid, pid, text FROM passages WHERE rowid IN ({placeholders})"
        rows = cur.execute(sql, tuple(need_rowids)).fetchall()
        rowid2pt = {rid: (pid, text) for (rid, pid, text) in rows}

        # 3) 準備重排序 (Reranker)
        batch_cand_ids = []
        batch_cand_texts = []
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
            continue

        # 4) 重排序
        flat_scores = reranker.predict(flat_pairs, batch_size=BATCH_Q*TOP_K)
        
        # 5) 儲存預處理結果
        for b, (q, (low, high)) in enumerate(zip(queries, idx_slices)):
            if low == high: continue
            
            scores = flat_scores[low:high]
            cand_ids = batch_cand_ids[b]
            cand_text = batch_cand_texts[b]
            
            # 儲存排序後的文件 (text, score, pid)
            reranked = sorted(zip(cand_text, scores, cand_ids), key=lambda x: x[1], reverse=True)
            
            precomputed_data.append({
                "qid": qids[b],
                "query": queries[b],
                # 原始的 384-D embedding 仍然被儲存，只是 RAGEnv._get_obs() 不使用它
                "query_emb": q_embs[b], 
                "gold_answer": gold_ans[b],
                "reranked_passages": reranked # 存 (text, score, pid)
            })

    print(f"Preprocessing complete. Created {len(precomputed_data)} precomputed training examples.")
    return precomputed_data

# --- 3. 主執行函式 (Main) ---
# (此函式 'main' 保持不變)

def main():
    argparser = argparse.ArgumentParser()
    # TODO: 請填入你作業的預設路徑
    argparser.add_argument("--data_folder", type=str, default="./data")
    argparser.add_argument("--passage_file", type=str, default="corpus.txt")
    argparser.add_argument("--index_folder", type=str, default="./vector_database")
    argparser.add_argument("--index_file", type=str, default="passage_index.faiss")
    argparser.add_argument("--sqlite_file", type=str, default="passage_store.db")
    argparser.add_argument("--train_data_path", type=str, default="./data/train.txt") # 使用 train.txt
    argparser.add_argument("--qrels_path", type=str, default="./data/qrels.txt")
    argparser.add_argument("--retriever_model_path", type=str, default="") # TODO: 填入你訓練好的 Retriever 路徑
    argparser.add_argument("--reranker_model_path", type=str, default="") # TODO: 填入你訓練好的 Reranker 路徑
    argparser.add_argument("--generator_model", type=str, default="Qwen/Qwen3-1.7B")
    argparser.add_argument("--rl_agent_save_path", type=str, default="rag_rl_agent.zip")
    argparser.add_argument("--total_timesteps", type=int, default=10000) # 訓練步數
    args = argparser.parse_args()

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # --- 載入預處理所需的模型 ---
    print("Loading models for preprocessing...")
    sqlite_path = f"{args.index_folder}/{args.sqlite_file}"
    conn = sqlite3.connect(sqlite_path)
    retriever = SentenceTransformer(args.retriever_model_path, device=DEVICE)
    reranker = CrossEncoder(args.reranker_model_path, device=DEVICE)
    index = faiss.read_index(os.path.join(args.index_folder, args.index_file))

    # --- 載入訓練資料 ---
    qrels_path = args.qrels_path
    
    def load_qrels_gold(qrels_path):
        with open(qrels_path, "r", encoding="utf-8") as f:
            qrels = json.load(f)
        qid2gold = {}
        for qid, pid2lab in qrels.items():
            gold = {pid for pid, lab in pid2lab.items() if str(lab) != "0"}
            qid2gold[qid] = gold
        return qid2gold

    train_data = []
    qid2gold = load_qrels_gold(qrels_path)
    with open(args.train_data_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            obj = json.loads(line)
            qid = obj.get("qid")
            if qid not in qid2gold: continue # 只使用有黃金答案的資料
            
            train_data.append({
                "qid": qid, 
                "query": obj.get("rewrite"), 
                "gold_answer": (obj.get("answer")).get("text", ""),
                "gold_pids": qid2gold.get(qid, set())
            })
    
    # --- 載入語料庫 ---
    corpus = {}
    with open(os.path.join(args.data_folder, args.passage_file), 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Reading Corpus"):
            data = json.loads(line)
            corpus[data['id']] = data['text']

    # --- 執行預處理 ---
    precomputed_data = preprocess_training_data(
        retriever, reranker, index, conn, train_data, qid2gold, corpus
    )
    
    # --- 釋放 VRAM ---
    print("Preprocessing finished. Releasing Retriever and Reranker from VRAM.")
    del retriever
    del reranker
    del index
    conn.close()
    gc.collect()
    torch.cuda.empty_cache()

    # --- 載入 RL 訓練所需的模型 ---
    print("Loading models for RL training (LLM and Sentence Scorer)...")
    tokenizer = AutoTokenizer.from_pretrained(args.generator_model)
    model = AutoModelForCausalLM.from_pretrained(args.generator_model, dtype="auto", device_map="auto")
    sentence_scorer = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=DEVICE)
    
    # --- 建立環境並開始訓練 ---
    print("Creating RAG Environment...")
    # PPO 會自動從 env 實例中讀取更新後的 action_space 和 observation_space
    env = RAGEnv(precomputed_data, model, tokenizer, sentence_scorer, device=DEVICE)
    env = DummyVecEnv([lambda: env]) # 包裝成 SB3 相容的環境

    print("Starting RL Agent training...")
    # 使用 PPO 演算法
    rl_agent = PPO(
        "MlpPolicy", 
        env, 
        verbose=1, 
        device="cpu", 
        n_steps=512,
        tensorboard_log="./rl_tensorboard_log/"
    )
    
    # 開始訓練
    rl_agent.learn(total_timesteps=args.total_timesteps)
    
    # 儲存訓練好的模型
    rl_agent.save(args.rl_agent_save_path)
    print(f"RL Agent training complete. Model saved to {args.rl_agent_save_path}")

if __name__ == "__main__":
    main()