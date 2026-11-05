"""Build new knowledge bases or open the artifacts produced by the notebooks."""
import json
import time
import uuid
from pathlib import Path
import pandas as pd
from config.settings import DATA_DIR, PROCESSED_DIR, EMBEDDING_MODELS
from src.data.wikipedia import build_dataset
from src.chunking.strategies import split_text
from src.embeddings.models import load_embedder, encode
from src.vectorstore.store import build_stores


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2), encoding='utf-8')


def build_knowledge_base(keywords, limit=2, method='recursive', model_label='MiniLM', progress=None):
    build_dir = PROCESSED_DIR / 'builds' / uuid.uuid4().hex[:12]
    build_dir.mkdir(parents=True)
    metadata, download_time = build_dataset(keywords, limit, build_dir, progress)
    model_name = EMBEDDING_MODELS[model_label]
    model = load_embedder(model_name)
    started = time.perf_counter()
    records = []
    for article in metadata.itertuples():
        text = (build_dir / 'cleaned' / article.file_name).read_text(encoding='utf-8')
        for i, chunk in enumerate(split_text(text, method, model=model)):
            records.append({'chunk_id': f'{article.article_id}_{i:05d}',
                'article_id': article.article_id, 'title': article.title,
                'source_url': article.url, 'category': article.category,
                'chunk_text': chunk, 'chunk_length': len(chunk), 'chunk_index': i})
    chunks = pd.DataFrame(records)
    chunk_time = time.perf_counter() - started
    if chunks.empty:
        raise ValueError('No chunks generated.')
    chunks.to_csv(build_dir / 'chunk_metadata.csv', index=False)
    started = time.perf_counter()
    vectors = encode(model, chunks.chunk_text.tolist())
    embedding_time = time.perf_counter() - started
    import numpy as np
    np.save(build_dir / 'embeddings.npy', vectors)
    started = time.perf_counter()
    build_stores(build_dir / 'vector_db', chunks, vectors)
    settings = {'model': model_name, 'method': method, 'articles': len(metadata),
                'chunks': len(chunks), 'dimensions': int(vectors.shape[1]),
                'download_seconds': download_time, 'chunking_seconds': chunk_time,
                'embedding_seconds': embedding_time, 'index_seconds': time.perf_counter() - started}
    save_json(build_dir / 'settings.json', settings)
    # Activate only after all stages succeed. Prior data remains intact.
    pointer = PROCESSED_DIR / 'active_build.json'
    temporary = pointer.with_suffix('.tmp')
    save_json(temporary, {'directory': str(build_dir.relative_to(DATA_DIR))})
    temporary.replace(pointer)
    return settings


def active_artifacts():
    pointer = PROCESSED_DIR / 'active_build.json'
    if pointer.exists():
        directory = DATA_DIR / json.loads(pointer.read_text())['directory']
        chunks = pd.read_csv(directory / 'chunk_metadata.csv')
        metadata = pd.read_csv(directory / 'metadata.csv')
        settings = json.loads((directory / 'settings.json').read_text())
        return directory, directory / 'vector_db', chunks, metadata, settings
    chunk_path = PROCESSED_DIR / 'chunk_metadata' / 'chunk_metadata.csv'
    if not chunk_path.exists():
        raise FileNotFoundError('Build a knowledge base first, or copy your existing data folder into this project.')
    chunks = pd.read_csv(chunk_path)
    metadata = pd.read_csv(DATA_DIR / 'metadata' / 'metadata.csv')
    metadata = metadata.rename(columns={'Article Name': 'title', 'URL': 'url', 'Category': 'category',
        'Word Count': 'word_count', 'Character Count': 'character_count', 'Download Date': 'download_date'})
    required = {'chunk_id', 'article_id', 'title', 'source_url', 'chunk_text'}
    if not required.issubset(chunks.columns) or chunks.chunk_id.duplicated().any() or chunks.chunk_text.isna().any():
        raise ValueError('Invalid chunk metadata. Rebuild the knowledge base.')
    return PROCESSED_DIR, PROCESSED_DIR / 'vector_db', chunks, metadata, {
        'model': EMBEDDING_MODELS['MiniLM'], 'method': 'semantic',
        'articles': int(chunks.article_id.nunique()), 'chunks': len(chunks)}
