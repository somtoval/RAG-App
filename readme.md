# 🧠📚 Modular RAG Pipeline — Powered by FastAPI + LangChain

Welcome to my **RAG (Retrieval-Augmented Generation)** project!  
This is a modular and scalable system designed to accept document inputs, allow custom system prompts, and retrieve information with memory — all via a clean FastAPI interface.

> ⚙️ Built for speed, flexibility, and continuous improvement.

---

## 🚀 Features

- 🗂️ Upload documents (PDFs)
- 🧩 Chunk documents and embed them using HuggingFace
- 🧠 Store vectorized documents with FAISS
- 🧾 Customizable system prompt (coming soon)
- 🤖 Powered by Groq + LLaMA 3 for lightning-fast LLM inference
- 🔁 Retrieval QA over your private documents
- 🖥️ FastAPI-ready for production deployment

---

## ✅ Current Progress

I’ve completed a working **RAG pipeline** with:

- LangChain + Groq LLM (LLaMA 3)
- HuggingFace sentence-transformers for embeddings
- FAISS for vector storage
- Simple document upload and question answering

```python
from pathlib import Path
from rag_pipeline import RagPipeline

pipeline = RagPipeline()
qa_chain = pipeline.run_pipeline(Path("example.pdf"))
response = qa_chain.run("What is this document about?")
print(response)
