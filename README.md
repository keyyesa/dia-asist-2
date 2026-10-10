# DiaAssist AI — Agentic-RAG Clinical Decision Support for Diabetes

A multi-agent Retrieval-Augmented Generation (RAG) system for diabetes knowledge retrieval and
lab-report interpretation. It runs **fully offline** (Ollama + local embeddings + FAISS) or with
the Gemini cloud API.

> ⚠️ Research/academic prototype. Not a medical device; never use it for real diagnosis or
> treatment decisions without qualified professional oversight.

---

## 1. Quick start

```bash
python -m venv venv
venv\Scripts\activate            # Windows   (macOS/Linux: source venv/bin/activate)
pip install -r requirements.txt

ollama pull llama3.1:8b          # one-time, needs internet (offline afterwards)
streamlit run app.py
```

Then in the app: **📚 Knowledge Base → upload documents → they are trained → 💬 Chat**.

The first run downloads the small embedding model (`all-MiniLM-L6-v2`) once; after that
nothing needs the internet.

## 2. How "training" works

A RAG chatbot is not trained like a neural network. *Training* means: read your documents →
split into chunks → embed every chunk → store the vectors in a FAISS index. A question is then
answered by searching that index (milliseconds) instead of re-reading the documents, and only
the best-matching chunks are sent to the LLM. That is why answers are fast.

| Action | What happens |
|---|---|
| Upload a file (auto-train **on**, default) | File is saved and trained immediately, with a progress bar |
| Upload a file (auto-train **off**) | File is saved; it is **not used** until you press **🚀 Train** |
| **🚀 Train** (or sidebar *Train now*) | Trains only new / changed files, drops deleted ones, reuses the rest |
| **♻️ Full retrain** | Rebuilds the entire index from scratch |
| Delete a document | Removed from the folder; its knowledge is dropped on the next train |

Training is *incremental*: a `manifest.json` stores a SHA-256 hash per trained file, so adding
one PDF to a big knowledge base only embeds that PDF. A full retrain happens automatically if you
change the embedding model or chunk size. Index files are written atomically, so an interrupted
training can never corrupt the knowledge base.

Supported documents: **PDF, TXT, MD, DOCX** (scanned PDFs need OCR first).

## 3. The interface

A sidebar menu groups every task:

- **🏠 Dashboard** — status of documents, chunks, last training, LLM; guided quick start
- **💬 Chat** — streaming answers, sources + relevance, example questions, export, optional lab report
- **🧪 Lab Reports** — upload/paste a report, extract and classify HbA1c, glucose, lipids…
- **📚 Knowledge Base** — upload, train, delete, and test retrieval without the LLM
- **⚖️ Compare RAG** — Agentic-RAG vs a traditional baseline, side by side
- **📊 Evaluation** — run the test questions through both pipelines, download `results.csv`
- **⚙️ Settings** — switch backend/model, test the connection, tune chunking, save to `.env`
- **ℹ️ About**

Sidebar quick controls (always visible): **Response mode**, **top-k**, KB status, a **Train now**
button, LLM health, **Clear chat**.

### Fast vs Full mode

| | ⚡ Fast (default) | 🧠 Full agentic |
|---|---|---|
| Agents used | Retrieval + Response | Query Optimizer → Router → Retrieval/Report → Response → Verification |
| LLM calls | 1 (streamed word-by-word) | up to 5 |
| Best for | everyday questions | careful, verified answers |

Other speed-ups: Ollama keeps the model in RAM between questions (`OLLAMA_KEEP_ALIVE`), identical
questions are answered instantly from a cache, and a lab report is analysed once and cached.

### Lab reports vs. the knowledge base

These are two different things:

- **Knowledge Base** = shared reference material (guidelines, papers). Documents are *trained* and every answer can draw on them.
- **Your lab report** = personal data. Attach it in **Chat (📎)** or on the **Lab Reports** page. Questions such as
  *"is my report normal?"* or *"what is my RBG from my report?"* are answered from that report only.

How a report is read: values are found with tolerant text rules (handles `HbAlC` typos, `FBS/RBS` abbreviations,
mmol/L and mmol/mol units, messy table layouts, and ignores the "interpretation legend" thresholds), classified with
fixed reference ranges, and shown next to the lab's own reference range. The LLM only reads the report when the rules
find nothing. The answer step also receives the raw report text and the list of tests that are **not** in the report,
so asking for a value the report doesn't contain gets a clear "not reported" instead of a guess.

## 4. Configuration

Everything lives in `.env` (see `.env.example`) and most options are editable on the Settings
page. Key options: `LLM_BACKEND`, `OLLAMA_MODEL`, `OLLAMA_KEEP_ALIVE`, `OLLAMA_NUM_CTX`,
`EMBEDDING_MODEL`, `CHUNK_SIZE`, `CHUNK_OVERLAP`, `TOP_K`.

Smaller/faster models if `llama3.1:8b` is slow on your PC: `llama3.2:3b`, `phi3:mini`.

## 5. Command line

```bash
python ingest.py          # train new/changed documents
python ingest.py --full   # rebuild everything
python evaluation/evaluate.py
python -m unittest discover -s tests -v      # run the test-suite (no Ollama/internet needed)
```

## 6. Project structure

```
app.py                      entry point: navigation + sidebar
views/                      one file per page (chat, knowledge_base, lab_report, ...)
src/vectorstore.py          incremental training, manifest, FAISS search
src/documents.py            PDF / DOCX / TXT text extraction
src/llm_client.py           Ollama / Gemini wrapper (streaming, keep-alive)
src/orchestrator.py         Agentic-RAG pipeline (fast + full modes, caching)
src/baseline_rag.py         traditional RAG for comparison
src/agents/                 optimizer, router, retrieval, report analysis, generation, verification
evaluation/                 test questions + evaluation runner
tests/                      backend + UI tests
data/knowledge_base_docs/   your documents
knowledge_base/faiss_index/ the trained index (generated)
```

## 7. Troubleshooting

- **"Cannot connect to Ollama"** — start it (`ollama serve`) and check the URL in Settings.
- **"Model … is not installed"** — `ollama pull <model>`.
- **Embedding model can't load** — connect to the internet once so it downloads, then restart.
- **Answers ignore a new document** — it has not been trained yet; press **Train** (watch the sidebar status).
- **Scanned PDF skipped** — it has no text layer; OCR it first.

## 8. Deployment

Docker: `docker build -t diaassist-ai .` then `docker run -p 8501:8501 diaassist-ai`
(the image trains the bundled documents at build time; set `LLM_BACKEND`/`OLLAMA_BASE_URL` or
`GEMINI_API_KEY` as environment variables). Streamlit Community Cloud also works with the Gemini backend.

## 9. Known limitations

- Lab reports are patient-specific, so they are analysed per session and are *not* added to the shared knowledge base.
- Image-only lab reports need an OCR step (not included).
- Each question is answered independently (no multi-turn memory).
- The bundled documents are placeholders; replace them with real guidelines before drawing conclusions.
