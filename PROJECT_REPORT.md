# Dynamic AI Knowledge Assistant — report draft

## Objective
Build a reusable English Wikipedia knowledge base from user-defined topics and answer questions with source-grounded retrieval-augmented generation.

## Implementation
The Python pipeline searches Wikipedia, follows redirects, filters duplicate and empty articles, preserves provenance, normalizes text, and compares fixed, recursive, and semantic chunking. Sentence Transformers supplies MiniLM and MPNet embeddings. FAISS and ChromaDB store vectors. Retrieval supports semantic search, BM25 rank fusion, and optional cross-encoder reranking. A local Qwen model streams responses through a five-page Streamlit interface.

## Design choices
Stable Wikipedia page IDs preserve document identity. Separate build directories prevent stale records on rebuild. The active pointer changes only after indexing succeeds. Prompt budgeting limits input length. Conversation history is bounded and is not treated as factual evidence.

## Experiments — fill after execution
Add dataset keywords, article count, words, chunks, CPU/RAM details, dependency versions, chunking times, embedding times, database size, retrieval metrics, retrieval latency, response time, and generation evaluation. Paste exported tables here. State whether experiments used original cleaned text or text reconstructed from chunks.

## Limitations
Small local models can produce unsupported claims or imperfect citations. Source links show retrieved evidence, not proof of correctness. Topic search can find tangential articles. The short follow-up heuristic is limited. No calibrated answer-confidence model, automated faithfulness evaluator, scheduled refresh, or persistent cloud deployment is supplied. The preserved original notebooks still contain dataset-specific assumptions; use the new modules for dynamic builds.

## Conclusion — complete after validation
Describe what your measured results show, your chosen chunking/model/database configuration, and known failure cases. Add app screenshots, GitHub URL, and deployment URL.
