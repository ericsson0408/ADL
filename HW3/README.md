# ADL HW3 - RAG (Retrieval-Augmented Generation) System

This project implements a complete RAG (Retrieval-Augmented Generation) system for question answering tasks. The system consists of three main components:

1. **Retriever**: Uses `intfloat/multilingual-e5-small` model to quickly retrieve relevant documents from the corpus
2. **Reranker**: Uses `cross-encoder/ms-marco-MiniLM-L-12-v2` model to precisely re-rank the retrieved results
3. **Generator**: Uses `Qwen/Qwen3-1.7B` model to generate answers based on the retrieved documents

## System Architecture

```
Query → Retriever (Retrieve Top-K Documents) → Reranker (Re-rank) → Generator (Generate Answer) → Final Answer
```

## Environment Requirements

Please ensure all necessary packages are installed according to `requirements.txt`:

```bash
pip install -r requirements.txt
```

Main packages include:
- `transformers==4.56.1` - Hugging Face transformers library
- `torch==2.8.0` - PyTorch deep learning framework
- `sentence-transformers==5.1.0` - Sentence embedding models
- `faiss-gpu-cu12==1.12.0` - Vector similarity search
- `datasets==4.0.0` - Data processing
- `accelerate==1.10.1` - Model training acceleration

## Quick Start

### 1. Download Pre-trained Models

First, run `download.sh` to download the pre-trained retriever and reranker models:

```bash
bash download.sh
```

This will automatically download and extract the `models/` folder, containing:
- `models/retriever/` - Trained retriever model
- `models/reranker/` - Trained reranker model

### 2. Run Inference

Use `infer.sh` to perform inference:

```bash
bash infer.sh
```

**Inference Process:**
1. Load retriever model from `./models/retriever`
2. Load reranker model from `./models/reranker`
3. Use FAISS vector database `./vector_database` for fast retrieval
4. Read test data from `data/test_open.txt`
5. Generate answers and output results

**Parameter Description:**
- `--retriever_model_path`: Path to the retriever model
- `--reranker_model_path`: Path to the reranker model
- `--index_folder`: Path to the FAISS vector database
- `--test_data_path`: Path to the test data

Inference process executes:
1. **Retrieval**: Use retriever to retrieve Top-K (default 10) relevant documents
2. **Reranking**: Use reranker to re-rank and select Top-M (default 3) most relevant documents
3. **Generation**: Use selected documents as context for the generation model to produce the final answer

Please ensure `save_embeddings.py` and `inference_batch.py` are in the same folder.
Please ensure the `data` folder and its content `test_open.txt` are in the same folder.

## Training Process

### Training Retriever

Use `train_retriever.sh` to train the retriever model:

```bash
bash train_retriever.sh
```

**Training Details:**
- **Base Model**: `intfloat/multilingual-e5-small`
- **Training Data**: `./data/train.txt` (contains query-document pairs)
- **Corpus**: `./data/corpus.txt` (all candidate documents)
- **Label Data**: `./data/qrels.txt` (relevance labels)
- **Training Strategy**: Multiple Negatives Ranking Loss (MNRL)

**Output:**
- Trained model saved in `retriever/` directory


### Training Reranker

Training the reranker requires two steps, using `train_reranker.sh`:

```bash
bash train_reranker.sh
```

**Step 1: Generate Hard Negatives**
- Use the trained retriever to retrieve Top-K (default 50) documents from the corpus
- Mark non-relevant documents in the retrieval results as "hard negatives"
- Output to `./data/train_with_hard_negs.jsonl`

**Step 2: Train Reranker**
- **Base Model**: `cross-encoder/ms-marco-MiniLM-L-12-v2`
- **Training Data**: `./data/train_with_hard_negs.jsonl` (contains hard negatives)
- **Training Strategy**: Binary Classification with hard negatives

**Output:**
- Trained model saved in `reranker/` directory

## Data Format

### Training Data (`train.txt`)
JSONL format, each line contains:
```json
{
  "rewrite": "query text",
  "evidences": ["passage1", "passage2", ...],
  "retrieval_labels": [0, 1, 0, ...]
}
```

### Corpus (`corpus.txt`)
Plain text format, one document passage per line.

### Test Data (`test_open.txt`)
JSONL format, each line contains a question.

## Required Files

Ensure the following files are in the project directory:

- ✅ `utils.py` - Contains prompt generation functions for inference:
  - `get_inference_system_prompt()`: Generate system prompt
  - `get_inference_user_prompt()`: Generate user prompt
  - `parse_generated_answer()`: Parse generated answer

- ✅ `requirements.txt` - List of all required packages

- ✅ `inference_batch.py` - Batch inference script

- ✅ `retriever.py` - Retriever training script

- ✅ `reranker.py` - Reranker training script

- ✅ `gen_hard_neg.py` - Script to generate hard negatives



## Technical Details

### Retriever
- Uses bi-encoder architecture, encoding query and document into the same vector space
- Uses MNRL loss during training, optimizing both positive sample similarity and negative sample separation
- Uses FAISS to accelerate vector retrieval during inference

### Reranker
- Uses cross-encoder architecture, processing query-document pairs together
- Introduces hard negatives to improve model discrimination ability
- Provides more precise relevance scores

### Generator
- Generates answers based on retrieved context
- Uses specific prompt templates to guide model responses
- Responds with "CANNOTANSWER" if the answer is not found in the context

## FAQ

**Q: How to adjust the number of retrieved documents?**
A: Modify the `TOP_K` and `TOP_M` parameters in `inference_batch.py`.

**Q: How much GPU memory is required for training?**
A: Retriever requires approximately 8GB, Reranker requires approximately 6GB.

**Q: How to use your own dataset?**
A: Prepare `train.txt`, `corpus.txt`, and `qrels.txt` according to the data format, then run the training scripts.

## References

- [Sentence Transformers Documentation](https://www.sbert.net/)
- [FAISS: A Library for Efficient Similarity Search](https://github.com/facebookresearch/faiss)
- [Qwen Models](https://huggingface.co/Qwen)


Requirement.txt:

transformers==4.56.1
torch==2.8.0 
datasets==4.0.0
tqdm==4.67.1
faiss-gpu-cu12==1.12.0
sentence-transformers==5.1.0
python-dotenv==1.1.1
accelerate==1.10.1
gdown


tokenizers==0.22.1
safetensors==0.6.2
scikit-learn==1.7.2
stable-baselines3==2.7.0
gymnasium==1.0.0
faiss-gpu==1.12.0
sentence-transformers==5.1.0
numpy==1.26.4
pandas==2.3.3
pyarrow==21.0.0
scipy==1.16.2
matplotlib-base==3.10.7

