"""Shared settings. Paths work regardless of the terminal's current directory."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / 'data'
PROCESSED_DIR = DATA_DIR / 'processed'
EMBEDDING_MODELS = {
    'MiniLM': 'sentence-transformers/all-MiniLM-L6-v2',
    'MPNet': 'sentence-transformers/all-mpnet-base-v2',
}
GENERATION_MODEL = 'Qwen/Qwen2.5-0.5B-Instruct'
RERANK_MODEL = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
COLLECTION_NAME = 'wiki_chunks_minilm'
