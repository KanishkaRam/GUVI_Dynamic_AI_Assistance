"""Both databases share the same row order and normalized embeddings."""
import numpy as np
from config.settings import COLLECTION_NAME


def build_stores(directory, chunks, embeddings):
    import faiss
    import chromadb
    if len(chunks) != len(embeddings) or not len(chunks):
        raise ValueError('Chunk and embedding counts must match and be nonzero.')
    directory.mkdir(parents=True, exist_ok=True)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, str(directory / 'index.faiss'))
    client = chromadb.PersistentClient(path=str(directory / 'chroma'))
    # Each build uses a unique directory, preventing stale records.
    collection = client.get_or_create_collection(COLLECTION_NAME,
        metadata={'hnsw:space': 'cosine'})
    for start in range(0, len(chunks), 250):
        batch = chunks.iloc[start:start + 250]
        collection.upsert(ids=batch.chunk_id.tolist(), documents=batch.chunk_text.tolist(),
            embeddings=embeddings[start:start + len(batch)].tolist(),
            metadatas=[{'title': str(r.title), 'source_url': str(r.source_url),
                        'article_id': str(r.article_id)} for r in batch.itertuples()])
    return index.ntotal, collection.count()


def vector_search(directory, chunks, query_vector, backend, top_k):
    import faiss
    import chromadb
    top_k = min(top_k, len(chunks))
    if backend == 'FAISS':
        path = directory / 'index.faiss'
        if not path.exists():
            path = directory / 'faiss' / 'minilm_faiss.index'
        index = faiss.read_index(str(path))
        if index.ntotal != len(chunks):
            raise ValueError('FAISS row count differs from chunk metadata. Rebuild the index.')
        scores, positions = index.search(query_vector, top_k)
        return [(str(chunks.iloc[int(i)].chunk_id), float(s)) for i, s in zip(positions[0], scores[0]) if i >= 0]
    if backend != 'ChromaDB':
        raise ValueError('Unknown database.')
    client = chromadb.PersistentClient(path=str(directory / 'chroma'))
    collection = client.get_collection(COLLECTION_NAME)
    result = collection.query(query_embeddings=query_vector.tolist(), n_results=top_k,
                              include=['distances'])
    return list(zip(result['ids'][0], [1 - d for d in result['distances'][0]]))
