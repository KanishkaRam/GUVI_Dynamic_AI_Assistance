import unittest
from src.data.wikipedia import clean_text, validate_keywords
from src.chunking.strategies import split_text
from src.retrieval.search import reciprocal_rank_fusion
from src.evaluation.metrics import retrieval_metrics, token_f1


class CoreTests(unittest.TestCase):
    def test_keywords_case_duplicates(self):
        with self.assertRaises(ValueError):
            validate_keywords(['AI', 'ai', 'ML', 'NLP', 'CV'])

    def test_unicode_cleaning(self):
        self.assertEqual(clean_text('  α\t beta\x00\r\n\n\nEnd '), 'α beta\n\nEnd')

    def test_fixed_coverage_and_no_redundant_tail(self):
        self.assertEqual(split_text('abcdefghij', 'fixed', 6, 2), ['abcdef', 'efghij'])
        self.assertEqual(split_text('', 'fixed'), [])
        with self.assertRaises(ValueError):
            split_text('abc', 'fixed', 5, 5)

    def test_article_metrics_deduplicate(self):
        result = retrieval_metrics(['a', 'a', 'b'], ['b'], 2)
        self.assertEqual(result['mrr_at_k'], 0.5)
        self.assertEqual(result['precision_at_k'], 0.5)
        self.assertEqual(result['recall_at_k'], 1)

    def test_fusion(self):
        self.assertEqual(reciprocal_rank_fusion([['a', 'b'], ['b', 'c']])[0], 'b')

    def test_reference_f1(self):
        self.assertEqual(token_f1('Machine learning [1]', 'Machine learning'), 1)
        self.assertEqual(token_f1('unrelated', 'machine learning'), 0)


if __name__ == '__main__':
    unittest.main()
