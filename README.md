# Twitter API Documentation Semantic Search (Colab Notebook Version)

This project provides a **semantic search engine** over the official **Postman Twitter API documentation**, implemented entirely inside a **single Google Colab or Jupyter notebook cell**.

You can load, chunk, embed, index, and semantically search the entire documentation — all inside a notebook, with no external setup required.

---

## 🚀 Features

- **Runs fully inside Colab/Jupyter**
- **Single-cell pipeline** (clone → chunk → embed → index → search)
- Uses **Sentence Transformers (all-mpnet-base-v2)**
- Fast semantic retrieval using **FAISS**
- CLI-style queries via:

  ```!python search.py --query "tweet expansions" --top_k 5


  ```
