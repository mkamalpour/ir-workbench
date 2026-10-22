"""Exercise user actions against the actual Streamlit application."""
from pathlib import Path
import unittest
from streamlit.testing.v1 import AppTest


def widget(elements, label):
    return next(element for element in elements if element.label == label)


class InterfaceTests(unittest.TestCase):
    def setUp(self):
        self.app = AppTest.from_file(str(Path(__file__).with_name('app.py'))).run(timeout=30)
        self.assertFalse(self.app.exception)

    def test_search_batch_and_stale_config(self):
        self.app.text_input(key='query').set_value('coffee').run()
        widget(self.app.button, 'Search corpus').click().run()
        self.assertFalse(self.app.exception)
        self.assertEqual(self.app.session_state['search_result'][4][0]['id'], 'cafe')
        self.app.text_area(key='batch_text').set_value('unknownunmatched\ncoffee').run()
        widget(self.app.button, 'Run batch search').click().run()
        self.assertFalse(self.app.exception)
        self.assertGreater(len(self.app.session_state['batch'][1]), 1)
        self.app.checkbox(key='stem').uncheck().run()
        self.assertTrue(widget(self.app.button, 'Search corpus').disabled)
        self.app.button(key='build').click().run()
        self.assertFalse(self.app.exception)
        self.assertFalse(widget(self.app.button, 'Search corpus').disabled)

    def test_corpus_validation_preserves_work(self):
        self.app.text_area(key='paste').set_value('A new coffee shop.\nA university library.').run()
        widget(self.app.button, 'Replace corpus with imports').click().run()
        self.assertEqual(len(self.app.session_state['docs']), 2)
        self.assertTrue(widget(self.app.button, 'Search corpus').disabled)
        self.app.button(key='build').click().run()
        self.assertFalse(self.app.exception)
        self.assertEqual(len(self.app.session_state['index'].documents), 2)
        self.app.text_area(key='paste').set_value('').run()
        widget(self.app.button, 'Replace corpus with imports').click().run()
        self.assertTrue(self.app.error)
        self.assertEqual(len(self.app.session_state['index'].documents), 2)

    def test_setqa_results_follow_input(self):
        widget(self.app.button, 'Analyze stories').click().run()
        self.assertFalse(self.app.exception)
        self.assertEqual(self.app.session_state['qa'][3]['accuracy'], 1)
        widget(self.app.text_area, 'Numbered story').set_value('invalid story').run()
        self.assertFalse(any(element.label == 'Download submission CSV' for element in self.app.get('download_button')))
        widget(self.app.button, 'Analyze stories').click().run()
        self.assertTrue(self.app.error)
        self.assertFalse(self.app.exception)

    def test_evaluation_and_companion_training(self):
        widget(self.app.button, 'Evaluate rankings').click().run()
        self.assertFalse(self.app.exception)
        self.assertIn('summary', self.app.session_state['evaluation'][3])
        for kind in ('author', 'essay'):
            key = 'companion_' + kind
            self.app.checkbox(key=key + '_demo').check().run()
            self.app.button(key=key + '_train').click().run()
            self.assertFalse(self.app.exception)
            self.assertIn(key, self.app.session_state)
            self.app.text_area(key=key + '_text').set_value('A synthetic passage about a library.').run()
            if kind == 'author':
                self.app.text_area(key=key + '_other').set_value('Another passage about a library.').run()
            self.app.button(key=key + '_predict').click().run()
            self.assertFalse(self.app.exception)


if __name__ == '__main__':
    unittest.main()
