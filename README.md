# Dynamic AI Knowledge Assistant

A beginner-friendly GUVI capstone application that builds a knowledge base from English Wikipedia and answers questions with a local RAG pipeline. Original notebooks are preserved in `notebooks/`; their hard-coded paths and counts have not been rewritten. The app uses reusable modules instead.

## Start on Windows

Extract this ZIP into a new folder first. You can copy your existing `data` folder into this extracted project to reuse the notebook artifacts. Do not replace your original project until you have checked the new app.

Open a terminal in the extracted project root:

```powershell
py -3.12 -m venv venv
.\venv\Scripts\python.exe -m pip install -r requirements.txt
.\venv\Scripts\python.exe -m streamlit run app/main.py
```

On Linux/macOS, use `python3 -m venv venv`, then `venv/bin/python -m pip install -r requirements.txt` and `venv/bin/python -m streamlit run app/main.py`.

No API key is required. Wikipedia access and first-time Hugging Face model downloads require internet. CPU generation can be slow. Start with five keywords and one or two articles each. Optional reranking and MPNet download extra models and require additional memory. The dependency ranges are compatibility targets, not a tested lockfile; record your working environment with `python -m pip freeze > requirements-lock.txt` after local validation.

## Existing notebook artifacts

Without an active app build, the app looks for:

- `data/metadata/metadata.csv` (your Phase 3 column names are supported)
- `data/processed/chunk_metadata/chunk_metadata.csv`
- `data/processed/vector_db/chroma/` with collection `wiki_chunks_minilm`
- `data/processed/vector_db/faiss/minilm_faiss.index`

Copy the entire persistent Chroma directory, not only its SQLite file. These artifacts were not included in the uploaded archive, so compatibility with their actual bytes remains to be checked locally. The fallback assumes MiniLM embeddings and a cosine Chroma collection, as in your notebooks.

## Pages

1. Dataset Builder accepts at least five distinct keywords, downloads articles, removes duplicate/empty/disambiguation pages, cleans text, records provenance, chunks text, embeds it, and builds FAISS and ChromaDB.
2. Knowledge Base displays metadata and indexed text.
3. Chat Assistant offers five prompt tasks, streaming Qwen generation, recent conversation context, hybrid search, optional cross-encoder reranking, source excerpts, and retrieval relevance.
4. Analytics displays counts, chunk distribution, embedding/build statistics, disk size, and session timings.
5. Evaluation compares databases; runs three chunking strategies against two embedding models; exports results; and calculates reference-answer token F1.

New builds live under `data/processed/builds/<build-id>/`. Each includes its own `vector_db/`. `active_build.json` selects the active build only after it finishes successfully. Old builds remain available on disk. Notebook databases stay in their original location. The app does not cache Wikipedia downloads or automatically reuse downloads between builds.

## How RAG works

Wikipedia text → cleaning → chunks → normalized embeddings → databases → retrieval → prompt with evidence → local Qwen answer.

Hybrid search combines vector and BM25 rankings using reciprocal rank fusion. Optional reranking scores question/chunk pairs. Context selection uses a 3,000-token input budget, preserving full excerpts that fit; it does not implement a separate learned context-compression model. Recent messages help interpret follow-ups, and short follow-ups include the previous question during retrieval. This simple memory strategy can occasionally mix topics; clear chat when changing subjects.

Sources identify excerpts passed to the generator. The model is instructed to cite claims, but a small model can still omit or misuse inline citations. Check the retrieved excerpts manually. Retrieval similarity is not a calibrated confidence score and does not prove correctness. Qwen is retained from your existing notebook; the brief lists Llama/Gemma/Mistral, so confirm that an alternative open model is acceptable for your evaluation.

## Evaluation procedure

Create 15–25 questions spanning your articles. Include definitions, comparisons, paraphrases, multi-turn follow-ups, and questions outside the knowledge base. Use the page's CSV template and replace the sample question. The CSV columns are `question,article_id`; article IDs are shown in the page. Each row currently accepts one relevant article.

The metrics are article-level Hit@5, Precision@5, Recall@5, and truncated MRR@5 after deduplication. Database evaluation considers up to 30 retrieved chunks before deduplicating; this may yield fewer than five unique articles. Precision uses a denominator of five. Timing includes query embedding and retrieval, so it is not isolated database-engine latency. The six experiments rank all chunk embeddings with NumPy cosine similarity to compare representations independently of database choice. Model loading is excluded from embedding timing; rerun warm and cold measurements separately if needed.

Use newly built datasets for fair chunking comparisons: they retain exact cleaned article files. For legacy artifacts, comparison reconstructs article text from chunks, which can include repeated overlap; treat those results as exploratory.

For generated answers, calculate token F1 against a human reference and manually score factual correctness, coverage, and source support. Token overlap is a simple metric, not a faithfulness estimator. Do not label it RAGAS or claim an LLM-judge evaluation. Export actual results; no benchmark numbers are fabricated or supplied here.

## Checks

```powershell
.\venv\Scripts\python.exe -m unittest discover -s tests -v
.\venv\Scripts\python.exe -m compileall -q app config src tests
```

These checks cover pure logic, not live model quality. Run the complete acceptance checklist in `ACCEPTANCE_CHECKLIST.md` locally.

## Deployment

For a Streamlit-compatible host, use `app/main.py` as the entry point and `requirements.txt` for dependencies. CPU model loading requires enough memory and the host must allow Wikipedia/Hugging Face access. Hosted disk may be temporary: knowledge bases can disappear after restart unless persistent storage is configured. Start with a small dataset. Deployment has not been performed by this package. Do not upload `venv`, Python caches, or credentials to GitHub. A small educational single-user app is the intended scope; shared concurrent builds need session isolation before wider hosting.

## Submission material

See `PROJECT_REPORT.md` for an editable report outline. Fill in your measured results, screenshots, repository URL, and deployment URL before submission. A final presentation and an actual deployed application are still submission tasks, not completed deliverables in this ZIP.

## Primary references

- https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct
- https://qwen.readthedocs.io/en/v2.5/inference/chat.html
- https://cookbook.chromadb.dev/core/collections/
- https://www.mediawiki.org/wiki/API:Search
