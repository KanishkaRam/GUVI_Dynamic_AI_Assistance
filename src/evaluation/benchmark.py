"""Compare strategies and models using the same articles and labelled questions."""
import time
import pandas as pd
from config.settings import EMBEDDING_MODELS
from src.chunking.strategies import split_text
from src.embeddings.models import load_embedder, encode
from src.evaluation.metrics import retrieval_metrics


def compare_experiments(documents, questions):
    rows = []
    # Documents contain article_id and text; questions contain question and article_id.
    for method in ['fixed', 'recursive', 'semantic']:
        semantic_model = load_embedder(EMBEDDING_MODELS['MiniLM'])
        start = time.perf_counter()
        texts, ids = [], []
        for document in documents:
            parts = split_text(document['text'], method, model=semantic_model)
            texts.extend(parts)
            ids.extend([document['article_id']] * len(parts))
        chunk_time = time.perf_counter() - start
        for name, model_name in EMBEDDING_MODELS.items():
            model = load_embedder(model_name)
            start = time.perf_counter()
            vectors = encode(model, texts)
            embed_time = time.perf_counter() - start
            scores = []
            for question in questions:
                vector = encode(model, [question['question']])[0]
                order = (vectors @ vector).argsort()[::-1]
                ranked_articles = list(dict.fromkeys(ids[i] for i in order))
                scores.append(retrieval_metrics(ranked_articles, [question['article_id']], 5))
            rows.append({'method': method, 'model': name, 'chunks': len(texts),
                'chunking_seconds': chunk_time, 'embedding_seconds': embed_time,
                'embedding_MB': vectors.nbytes / 1024 ** 2,
                **pd.DataFrame(scores).mean().to_dict()})
    return pd.DataFrame(rows)
