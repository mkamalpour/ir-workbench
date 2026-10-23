import json
import math
import random
import unittest

from ir_workbench.core import Document, SearchIndex, Settings, demo_documents, evaluate_rankings, parse_documents, parse_qrels, tokenize


class RetrievalTests(unittest.TestCase):
    def test_inverted_equals_exhaustive_for_all_weightings(self):
        rng = random.Random(19)
        docs = [Document(str(i), ' '.join(rng.choices(['apple', 'pear', 'coffee', 'cake', 'tea'], k=rng.randint(1, 30)))) for i in range(40)]
        for weighting in ('smoothed', 'assignment1', 'assignment2'):
            index = SearchIndex(docs, Settings(stemming=False), weighting)
            for metric in ('cosine', 'euclidean', 'manhattan'):
                for query in ('apple coffee', 'missing', '', 'tea cake cake'):
                    self.assertEqual(index.search(query, 9, metric, 'inverted'), index.search(query, 9, metric, 'exhaustive'))

    def test_known_cosine_and_explanation(self):
        index = SearchIndex([Document('a', 'apple pear'), Document('b', 'apple'), Document('c', 'coffee')], Settings(stemming=False))
        hits = index.search('apple')
        self.assertEqual(hits[0]['id'], 'b')
        self.assertAlmostEqual(hits[0]['score'], 1)
        self.assertEqual(index.explain('coffee', 'b')[0]['contribution'], 0)

    def test_ties_and_single_document(self):
        index = SearchIndex([Document('z', 'apple'), Document('a', 'apple')])
        self.assertEqual([row['id'] for row in index.search('apple')], ['a', 'z'])
        self.assertAlmostEqual(SearchIndex([Document('a', 'apple')]).search('apple')[0]['score'], 1)

    def test_preprocessing_unicode_and_filters(self):
        config = Settings(stemming=False, stopwords=('the',))
        self.assertEqual(tokenize('The apples 12.5 https://example.com me@test.org سلام', config), ['apples', 'num', 'سلام'])
        self.assertEqual(tokenize('running', Settings()), ['run'])

    def test_validation(self):
        for docs in ([], [Document('a', 'text'), Document('a', 'other')], [Document('a', '')]):
            with self.assertRaises(ValueError):
                SearchIndex(docs)
        with self.assertRaises(ValueError):
            Settings(min_length=20, max_length=2)
        with self.assertRaises(ValueError):
            SearchIndex(demo_documents()).search('coffee', k=-1)

    def test_document_formats(self):
        self.assertEqual(parse_documents('a.csv', b'id,text\r\na,"hello, world"')[0], Document('a', 'hello, world'))
        self.assertEqual(parse_documents('a.jsonl', b'{"review_id":"r1","text":"coffee"}\n')[0].id, 'r1')
        self.assertEqual(parse_documents('a.txt', b'\xef\xbb\xbfhello')[0].text, 'hello')
        for filename, data in [('a.json', b'['), ('a.csv', b'wrong\nx'), ('a.txt', b'\xff'), ('a.json', b'[{"id":"a","text":null}]')]:
            with self.assertRaises(ValueError):
                parse_documents(filename, data)

    def test_metrics_penalize_short_and_missing_rankings(self):
        report = evaluate_rankings({'q': ['a']}, {'q': ['a', 'b'], 'missing': ['c']}, 5)
        self.assertEqual(report['rows'][0]['precision'], 0.2)
        self.assertEqual(report['rows'][0]['recall'], 0.5)
        self.assertEqual(report['rows'][1]['precision'], 0)
        self.assertEqual(report['summary']['map'], 0.25)
        perfect = evaluate_rankings({'q': ['a', 'b']}, {'q': ['a', 'b']}, 2)
        self.assertEqual(perfect['summary']['ndcg'], 1)

    def test_qrels_and_index_exports(self):
        self.assertEqual(parse_qrels(b'query_id,doc_id,relevance\nq,a,1\nq,b,0'), {'q': ['a']})
        with self.assertRaises(ValueError):
            parse_qrels(b'{"q": "a"}')
        index = SearchIndex(demo_documents())
        exported = json.loads(index.index_json())
        self.assertEqual(exported['coffe']['df'], 2)
        self.assertIn('cafe', exported['coffe']['postings'])


if __name__ == '__main__':
    unittest.main()
