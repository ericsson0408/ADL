import torch
import requests
from datasets import load_dataset
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# ==========================================
# 1. 配置設定 (Configuration)
# ==========================================

# 設定 Google API (請替換成您自己的)
GOOGLE_API_KEY = "YOUR_GOOGLE_API_KEY"
GOOGLE_CSE_ID = "YOUR_SEARCH_ENGINE_ID" # Custom Search Engine ID (CX)

# 設定 Llama 3.1 模型 (需 Hugging Face 權限)
MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"

# ==========================================
# 2. 工具函數: Google Search RAG
# ==========================================

def search_google(query, num_results=3):
    """
    利用 Google Custom Search API 獲取相關新聞背景
    """
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        'q': query,
        'key': GOOGLE_API_KEY,
        'cx': GOOGLE_CSE_ID,
        'num': num_results,
        'dateRestrict': 'm1', # 限制搜尋最近一個月的資料 (可選)
    }
    
    try:
        response = requests.get(url, params=params)
        results = response.json()
        
        context_text = ""
        if 'items' in results:
            for item in results['items']:
                title = item.get('title', '')
                snippet = item.get('snippet', '')
                context_text += f"- Title: {title}\n  Snippet: {snippet}\n"
        return context_text
    except Exception as e:
        print(f"Google Search Error: {e}")
        return ""

# ==========================================
# 3. 工具函數: Llama 3.1 生成器
# ==========================================

class LlamaEngine:
    def __init__(self, model_id):
        print("Loading Llama 3.1 model (4-bit quantization)...")
        
        # 使用 4-bit 量化以節省 VRAM
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4"
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto"
        )
        
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9
        )

    def generate_fake_news(self, real_news, context):
        """
        構建 Prompt 並調用 Llama 產生假新聞
        """
        title = real_news['title']
        content = real_news['text'][:1000] # 截取前1000字避免過長

        # 構建 Prompt (參考原本的 rewrite 邏輯)
        system_prompt = (
            "You are a sophisticated writer. Your task is to rewrite a real news story "
            "to introduce believable factual errors or alter key entities (names, locations, events) "
            "while maintaining the journalistic tone. "
            "The goal is to create a piece of 'Fake News' that is plausible enough to fool fact-checkers."
        )

        user_prompt = f"""
### Background Information (from Google Search):
{context}

### Original Real News:
Title: {title}
Content: {content}

### Task:
Please rewrite the news above. 
1. Use the Background Information to add realistic details but twist the main facts.
2. Keep the style professional.
3. Output format:\nTitle: [New Title]\nBody: [New Body]
"""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        outputs = self.pipe(prompt)
        generated_text = outputs[0]["generated_text"]
        
        # 簡單處理回傳字串，只取生成部分
        return generated_text.split("<|start_header_id|>assistant<|end_header_id|>")[-1].strip()

# ==========================================
# 4. 主程序
# ==========================================

def main():
    # --- A. Input: 真實新聞 Dataset ---
    # 使用 'cnn_dailymail' 作為替代方案，它包含大量真實新聞
    print("Loading Dataset (CNN/DailyMail)...")
    dataset = load_dataset("cnn_dailymail", "3.0.0", split="test[:5]") # 先拿5筆測試
    
    # --- B. Engine: 初始化 Llama ---
    llama = LlamaEngine(MODEL_ID)

    # --- C. 處理每一則新聞 ---
    for i, news_item in enumerate(dataset):
        print(f"\n{'='*20} Processing News {i+1} {'='*20}")
        
        real_title = news_item['article'].split('\n')[0] if 'title' not in news_item else news_item['title']
        # CNN dataset 的 title 有時在 article 裡，這裡做個簡單處理
        formatted_news = {
            'title': "Breaking News", # 若 dataset 沒 title 欄位，可自行調整提取邏輯
            'text': news_item['article']
        }
        
        print(f"Original News Snippet: {formatted_news['text'][:100]}")

        # --- D. Technique: RAG (Call Google) ---
        # 搜尋關鍵字：取文章前幾個字或摘要做搜尋
        search_query = formatted_news['text'][:50] 
        print(f"Searching Google for context: '{search_query}'...")
        
        rag_context = search_google(search_query)
        print(f"Retrieved {len(rag_context.splitlines())} lines of context.")

        # --- E. Output: 生成假新聞 ---
        fake_news = llama.generate_fake_news(formatted_news, rag_context)
        
        print(f"\n[Generated Fake News]:\n{fake_news}")
        print("-" * 50)

if __name__ == "__main__":
    main()