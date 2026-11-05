"""Article-level retrieval metrics and reference-answer token overlap."""
import re
from collections import Counter


def retrieval_metrics(retrieved, relevant, k):
    relevant = set(relevant)
    if not relevant or k < 1:
        raise ValueError('Provide relevant article IDs and positive k.')
    ranked = list(dict.fromkeys(retrieved))[:k]
    hits = [i for i, identifier in enumerate(ranked, 1) if identifier in relevant]
    return {'hit_at_k': float(bool(hits)), 'precision_at_k': len(hits) / k,
            'recall_at_k': len(hits) / len(relevant), 'mrr_at_k': 1 / hits[0] if hits else 0.0}


def token_f1(answer, reference):
    def tokens(text):
        return Counter(re.findall(r'\w+', re.sub(r'\[\d+\]', '', text.lower())))
    a, b = tokens(answer), tokens(reference)
    overlap = sum((a & b).values())
    if not a or not b or not overlap:
        return 0.0
    precision, recall = overlap / sum(a.values()), overlap / sum(b.values())
    return 2 * precision * recall / (precision + recall)
