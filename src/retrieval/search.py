import re
import time
from functools import lru_cache
import numpy as np
from config.settings import RERANK_MODEL
from src.vectorstore.store import vector_search


def tokenize(text):
    return re.findall(r'\w+', text.lower())


def reciprocal_rank_fusion(rankings, constant=60):
    scores = {}
    for ranking in rankings:
        for rank, identifier in enumerate(dict.fromkeys(ranking), 1):
            scores[identifier] = scores.get(identifier, 0) + 1 / (constant + rank)
    return sorted(scores, key=scores.get, reverse=True)


@lru_cache(maxsize=1)
def reranker():
    from sentence_transformers import CrossEncoder
    return CrossEncoder(RERANK_MODEL)


def retrieve(query, chunks, database, model_name, backend='ChromaDB', top_k=5,
             hybrid=True, rerank=False):
    if not query.strip() or top_k < 1:
        raise ValueError('Enter a question and a positive top_k.')
    from src.embeddings.models import load_embedder, encode
    from rank_bm25 import BM25Okapi
    started = time.perf_counter()
    vector = encode(load_embedder(model_name), [query])
    candidates = vector_search(database, chunks, vector, backend, min(len(chunks), max(20, top_k)))
    similarities = dict(candidates)
    ranking = [identifier for identifier, score in candidates]
    if hybrid:
        bm25 = BM25Okapi([tokenize(t) for t in chunks.chunk_text])
        lexical_scores = bm25.get_scores(tokenize(query))
        order = np.argsort(-lexical_scores)
        lexical = [str(chunks.iloc[i].chunk_id) for i in order[:max(20, top_k)] if lexical_scores[i] > 0]
        ranking = reciprocal_rank_fusion([ranking, lexical])
    lookup = chunks.set_index('chunk_id')
    if any(i not in lookup.index for i in ranking):
        raise ValueError('Database contains stale chunk IDs. Rebuild the knowledge base.')
    if rerank:
        scores = reranker().predict([(query, lookup.loc[i, 'chunk_text']) for i in ranking])
        ranking = [ranking[i] for i in np.argsort(-scores)]
    results = lookup.loc[ranking[:top_k]].reset_index().copy()
    results['similarity_score'] = results.chunk_id.map(similarities)
    results['rank'] = range(1, len(results) + 1)
    return results, time.perf_counter() - started
