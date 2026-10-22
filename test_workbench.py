import importlib.util
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parent

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

class WorkbenchTests(unittest.TestCase):
    def test_parser_rejects_partially_malformed_input(self):
        qa = load('qa', 'tools/setqa/solution.py')
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'story.txt'
            source.write_text('1 Daniel picked up the apple.\nbroken line\n2 What is Daniel holding?\n', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'Line 2'):
                qa.parse_file(source)

    def test_export_without_extension_preserves_submission(self):
        qa = load('qa', 'tools/setqa/solution.py')
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'submission'
            qa.write_submission([], str(output))
            self.assertEqual(output.read_text().strip(), 'textID,sortedAnswerList')
            self.assertTrue(output.with_name('submission_readable.csv').is_file())

    def test_question_uses_inventory_at_question_time(self):
        qa = load('qa', 'tools/setqa/solution.py')
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'story.txt'
            source.write_text('1 Daniel picked up the apple.\n2 What is Daniel holding?\n3 Daniel dropped the apple.\n4 What is Daniel holding?\n', encoding='utf-8')
            results = qa.run(source)
            self.assertEqual(len(results), 2)
            self.assertNotEqual(results[0]['predicted_answer'], '-1')
            self.assertEqual(results[1]['predicted_answer'], '-1')

    def test_index_rebuild_and_retrieval(self):
        ir = load('inverted', 'tools/inverted/app.py')
        index = ir.InvertedIndex()
        index.build([['apple'], ['pear'], []])
        engine = ir.RetrievalEngine(index, 3)
        self.assertEqual(engine.query(['apple'])[0][0], 0)
        self.assertEqual(engine.query([]), [])
        self.assertEqual(engine.query(['unknown']), [])
        index.build([['orange']])
        self.assertNotIn('apple', index.index)

    def test_preprocessing_and_cosine(self):
        ir = load('tfidf', 'tools/tfidf/app.py')
        processor = ir.Preprocessor({'the'}, stemming=False)
        self.assertEqual(processor.preprocess('The apple!'), ['apple'])
        model = ir.TFIDFModel().fit([['apple'], ['pear'], ['plum']])
        results = ir.Retrieval(model).retrieve(model.transform(['apple']))
        self.assertEqual(results[0][0], 0)

if __name__ == '__main__':
    unittest.main()
