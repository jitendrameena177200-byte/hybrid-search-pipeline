# ⚡ Hybrid Search Pipeline

A high-performance Information Retrieval (IR) and RAG pre-retrieval pipeline combining **Sparse Keyword Matching (BM25)** with **Dense Semantic Embeddings (SentenceTransformers)** fused via **Reciprocal Rank Fusion (RRF)**.

---

## 🎯 Features

* **Sparse Retrieval:** Exact keyword and domain-specific token matching powered by `BM25Okapi`.
* **Dense Retrieval:** Deep semantic contextual understanding powered by `SentenceTransformers` (`all-MiniLM-L6-v2`).
* **Reciprocal Rank Fusion (RRF):** Parameter-free rank aggregation method ensuring stable, scale-invariant fusion without manual score normalization.
* **RAG Ready:** Easily adaptable as a retriever backend for LangChain, LlamaIndex, or custom LLM context augmentation pipelines.

---

## 🏗️ Architecture Overview

```text
                ┌────────────────┐
                │   User Query   │
                └───────┬────────┘
                        │
            ┌───────────┴───────────┐
            ▼                       ▼
    ┌───────────────┐       ┌───────────────┐
    │ BM25 (Sparse) │       │ Dense Vectors │
    └───────┬───────┘       └───────┬───────┘
            │                       │
            │ Top-K                 │ Top-K
            └───────────┬───────────┘
                        ▼
            ┌───────────────────────┐
            │  Reciprocal Rank      │
            │     Fusion (RRF)      │
            └───────────┬───────────┘
                        ▼
            ┌───────────────────────┐
            │  Fused Top-N Results  │
            └───────────────────────┘
```

---

## 🚀 Quickstart

### 1. Clone the Repository
```bash
git clone https://github.com/jitendrameena177200-byte/hybrid-search-pipeline.git
cd hybrid-search-pipeline
```

### 2. Set Up Environment & Install Dependencies
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Run the Pipeline
```bash
python main.py
```

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for details.
