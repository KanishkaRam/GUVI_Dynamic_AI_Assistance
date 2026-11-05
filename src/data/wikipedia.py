"""Collect plain-text English Wikipedia articles and preserve their provenance."""
import hashlib
import re
import time
import unicodedata
from datetime import datetime, timezone
from urllib.parse import quote
import pandas as pd


def clean_text(text):
    text = unicodedata.normalize('NFC', text).replace('\r\n', '\n').replace('\r', '\n')
    text = ''.join(c for c in text if unicodedata.category(c) != 'Cc' or c in '\n\t')
    text = '\n'.join(re.sub(r'[ \t]+', ' ', line).strip() for line in text.splitlines())
    return re.sub(r'\n{3,}', '\n\n', text).strip()


def validate_keywords(keywords):
    unique = {}
    for keyword in keywords:
        if isinstance(keyword, str) and keyword.strip():
            unique.setdefault(keyword.strip().casefold(), keyword.strip())
    if len(unique) < 5:
        raise ValueError('Enter at least five different keywords.')
    return list(unique.values())


def request_wikipedia(params):
    import requests
    last_error = 'Request failed'
    for attempt in range(5):
        try:
            response = requests.get('https://en.wikipedia.org/w/api.php', params={
                'format': 'json', **params}, headers={
                'User-Agent': 'DynamicAIKnowledgeAssistant/1.0 (GUVI educational project)'}, timeout=30)
            if response.status_code == 429 or response.status_code >= 500:
                last_error = f'HTTP {response.status_code}'
                time.sleep(min(30, 2 ** (attempt + 1)))
                continue
            response.raise_for_status()
            payload = response.json()
            if 'error' in payload:
                raise ValueError(str(payload['error']))
            return payload
        except (requests.RequestException, ValueError) as error:
            last_error = str(error)
            if attempt < 4:
                time.sleep(min(16, 2 ** attempt))
    raise RuntimeError(f'Wikipedia request failed: {last_error}')


def build_dataset(keywords, limit, directory, progress=None):
    """Write into a new build directory; failed builds never replace existing data."""
    keywords = validate_keywords(keywords)
    if not 1 <= limit <= 20:
        raise ValueError('Article limit must be between 1 and 20.')
    raw = directory / 'raw'
    cleaned = directory / 'cleaned'
    raw.mkdir(parents=True, exist_ok=True)
    cleaned.mkdir(parents=True, exist_ok=True)
    candidates = {}
    errors = []
    started = time.perf_counter()
    for keyword in keywords:
        try:
            payload = request_wikipedia({'action': 'query', 'list': 'search',
                'srsearch': keyword, 'srlimit': limit, 'srnamespace': 0})
            for result in payload.get('query', {}).get('search', []):
                candidates.setdefault(result['title'], keyword)
        except RuntimeError as error:
            errors.append({'title': keyword, 'error': str(error)})
    records, hashes, pages = [], set(), set()
    for number, (title, category) in enumerate(candidates.items(), 1):
        try:
            payload = request_wikipedia({'action': 'query', 'prop': 'extracts|info|pageprops',
                'explaintext': 1, 'inprop': 'url', 'redirects': 1, 'titles': title})
            page = next(iter(payload.get('query', {}).get('pages', {}).values()), {})
            if 'missing' in page or 'disambiguation' in page.get('pageprops', {}):
                continue
            text = page.get('extract', '').strip()
            clean = clean_text(text)
            if not clean:
                continue
            digest = hashlib.sha256(clean.encode()).hexdigest()
            if digest in hashes or page.get('pageid') in pages:
                continue
            hashes.add(digest)
            pages.add(page.get('pageid'))
            article_id = f"WIKI_{page['pageid']}"
            filename = article_id + '.txt'
            (raw / filename).write_text(text, encoding='utf-8')
            (cleaned / filename).write_text(clean, encoding='utf-8')
            records.append({'article_id': article_id, 'title': page['title'],
                'url': page.get('fullurl', 'https://en.wikipedia.org/wiki/' + quote(page['title'].replace(' ', '_'))),
                'category': category, 'file_name': filename, 'word_count': len(clean.split()),
                'character_count': len(clean), 'content_hash': digest,
                'download_date': datetime.now(timezone.utc).isoformat()})
        except (RuntimeError, KeyError, OSError) as error:
            errors.append({'title': title, 'error': str(error)})
        if progress:
            progress(number / max(1, len(candidates)), title)
        time.sleep(0.5)
    pd.DataFrame(errors, columns=['title', 'error']).to_csv(directory / 'error_log.csv', index=False)
    if not records:
        raise ValueError('No usable articles downloaded. Check the error log and keywords.')
    metadata = pd.DataFrame(records)
    metadata.to_csv(directory / 'metadata.csv', index=False)
    return metadata, time.perf_counter() - started
