from functools import lru_cache


@lru_cache(maxsize=2)
def load_embedder(name):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(name)


def encode(model, texts):
    return model.encode(texts, batch_size=32, normalize_embeddings=True,
                        convert_to_numpy=True, show_progress_bar=False).astype('float32')
