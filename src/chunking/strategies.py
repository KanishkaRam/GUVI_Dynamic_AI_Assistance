"""Three readable chunking strategies; sizes are measured in characters."""
import re
import numpy as np


def split_text(text, method='recursive', size=1000, overlap=200, model=None):
    if size <= 0 or not 0 <= overlap < size:
        raise ValueError('Require size > 0 and 0 <= overlap < size.')
    if not text.strip():
        return []
    if method == 'fixed':
        chunks = []
        start = 0
        while start < len(text):
            chunks.append(text[start:start + size].strip())
            if start + size >= len(text):
                break
            start += size - overlap
        return [c for c in chunks if c]
    if method == 'recursive':
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        return RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap,
            separators=['\n\n', '\n', '. ', ' ', '']).split_text(text)
    if method != 'semantic' or model is None:
        raise ValueError('Semantic chunking requires an embedding model.')
    units = [s.strip() for s in re.split(r'\n\s*\n|(?<=[.!?])\s+', text) if s.strip()]
    vectors = model.encode(units, normalize_embeddings=True, convert_to_numpy=True)
    similarities = np.sum(vectors[:-1] * vectors[1:], axis=1)
    threshold = float(np.percentile(similarities, 20)) if len(similarities) else -1
    chunks, current = [], ''
    for i, unit in enumerate(units):
        if current and len(current) + len(unit) + 1 > size:
            chunks.append(current)
            current = ''
        if len(unit) > size:
            if current:
                chunks.append(current)
                current = ''
            chunks.extend(split_text(unit, 'fixed', size, 0))
            continue
        current = (current + ' ' + unit).strip()
        if i < len(similarities) and similarities[i] <= threshold:
            chunks.append(current)
            current = ''
    if current:
        chunks.append(current)
    return chunks
