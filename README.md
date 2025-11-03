# 🩺 Medical RAG Chatbot — Project Overview and Runbook 📚

![Project Flow](./assest/projectflow.jpg)

### 1️⃣ What this project does

* 💬 Retrieval-Augmented Generation (RAG) chatbot that answers **medical questions** using your local PDF corpus 📄.  
* 🚀 Two ways to use it:  
  * 🖥️ CLI (terminal): fast, no UI  
  * 🌐 Chainlit UI: web interface at a local URL

### 2️⃣ Key components (files/folders)

* 📂 `data/`: Put your source PDFs here  
* 🗄️ `vectorstore/db_faiss/`: FAISS vector index created from PDFs  
* 🛠️ `ingest.py`: Builds/refreshes the FAISS index from `data/`  
* 🖥️ `cli.py`: Runs full RAG pipeline from terminal  
* 🌐 `model.py`: Chainlit UI app (starts local web server)  
* 📦 `requirements.txt`: Python dependencies  
* 🐍 `venv311/`: Python 3.11 virtual environment used to run the app  

### 3️⃣ Models used

* 📚 Embeddings: `sentence-transformers/all-MiniLM-L6-v2` (Hugging Face)  
* 🧠 Vector DB: `FAISS` (stored at `vectorstore/db_faiss/`)  
* 🤖 Generator (LLM): `TheBloke/Llama-2-7B-Chat-GGML` via `ctransformers` (CPU-friendly quantized model)  

### 4️⃣ End-to-end data flow

1. 📄 Load documents (PDFs) from `data/`  
2. ✂️ Split docs into chunks & embed using `all-MiniLM-L6-v2`  
3. 💾 Store chunks + embeddings in FAISS vector DB  
4. 🔍 Retriever pulls top-k relevant chunks from FAISS at query time  
5. 📝 Prompt template combines retrieved context + question  
6. 🤖 LLM (ctransformers) generates concise answer  
7. 🌐 UI shows brief citations/snippets from top retrieved chunks  

### 5️⃣ How to run (recommended: Python 3.11 venv)

Activate venv311 (already in repo):

```bash
cd D:\Research-Work@abhaynautiyal\GenAI-Project\medical_chatbot-main
./venv311/Scripts/activate
```

Install dependencies (if needed):

```bash
python -m pip install -U pip
python -m pip install chainlit langchain langchain_community langchain-huggingface sentence_transformers faiss_cpu ctransformers
```

Make sure FAISS index exists (`vectorstore/db_faiss/`). To rebuild see section 8.

Run the UI (Chainlit):

```bash
set CHAINLIT_NO_WATCH=1
python -m chainlit run model.py --host 127.0.0.1 --port 8012
# Open http://127.0.0.1:8012 in your browser
```

Run CLI (terminal):

```bash
python cli.py "What is hypertension?"
```

### 6️⃣ Important notes for first run

* ⏳ First query can take 30-90 seconds due to model loading/caching
* 🖥️ If UI only shows welcome text, check terminal logs for `[Chainlit] Question:` and wait

### 7️⃣ Prompting 

* Clinical, guideline-based questions work best. Examples:
  * What are common symptoms of meningitis?
  * Hi, I've had a sore throat and hoarse voice for two days. Could it be laryngitis?
  * My grandmother suddenly can't move one side of her body — what should we do?
  * What is the difference between acute and chronic kidney disease?
  * I often feel burning pain in my upper stomach after meals. Is that gastritis or an ulcer?
  * First-line therapy for resistant hypertension on ACEi + CCB + thiazide?
  * GOLD escalation criteria after COPD exacerbations?

### 8️⃣ Rebuilding the FAISS index (optional)

* 🛠️ Use `ingest.py` (requires `langchain-text-splitters` if import errors occur)

Rebuild index:

```bash
python ingest.py
```

Reads PDFs, creates embeddings, saves FAISS to `vectorstore/db_faiss/`

### 9️⃣ Troubleshooting

* ⚠️ Port in use (error 10048):
  * Kill existing server or change port
  * Find PID: `netstat -ano | findstr :8012`
  * Kill: `taskkill /PID <PID> /F`
  * Or change port: `--port 8013`

* 🖥️ UI shows only welcome text:
  * Wait for terminal logs; first request may be slow
  * Confirm running in `venv311` (Python 3.11)

* 🖥️ CLI works but UI doesn't:
  * Check both use same FAISS path `vectorstore/db_faiss`
  * Confirm required packages installed in venv311

### 🔟 Project structure recap

```
medical_chatbot-main/
  data/                        # 📄 Your PDFs
  vectorstore/
    db_faiss/                  # 🧠 FAISS index (index.faiss, index.pkl)
  cli.py                       # 💻 CLI entry point
  model.py                     # 🌐 Chainlit UI entry
  ingest.py                    # 🛠️ Builds FAISS index from PDFs
  requirements.txt             # 📦 Dependencies
  chainlit.md                  # 📖 UI intro/help (optional)
  venv311/                     # 🐍 Python 3.11 environment (recommended)
  assets/
    projectflow.jpg            # 🖼️ Project flow image
```

### 🕒 Daily usage quickstart

1. 🐍 Activate venv311
2. 🚀 Start UI on free port (e.g., 8012)
3. 💬 Ask your clinical question in UI
4. 🔄 If UI busy, test same question via CLI
