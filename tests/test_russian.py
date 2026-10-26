import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

# Corpus tooling must be verifiable without installing the training framework.
spec = importlib.util.spec_from_file_location('russian_data', Path(__file__).parents[1] / 'guppylm/russian_data.py')
ru = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ru)


class CorpusTests(unittest.TestCase):
    def test_reproducible_disjoint_and_russian(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            self.assertEqual(ru.build(a), ru.build(b))
            splits = {}
            for name in ('train', 'eval'):
                rows = [json.loads(line) for line in (Path(a) / f'{name}.jsonl').read_text(encoding='utf-8').splitlines()]
                self.assertTrue(rows)
                self.assertEqual(len(rows), len({r['text'] for r in rows}))
                splits[name] = {r['family'] for r in rows}
                for row in rows:
                    text = row['text']
                    self.assertEqual(text.count('<|im_end|>'), 3)
                    answer = text.split('<|im_start|>assistant\n')[1].split('<|im_end|>')[0]
                    self.assertTrue(all('а' <= c.lower() <= 'я' or c.lower() == 'ё' for c in answer if c.isalpha()))
            self.assertFalse(splits['train'] & splits['eval'])


class LanguageTests(unittest.TestCase):
    def test_guard(self):
        path = Path(__file__).parents[1] / 'guppylm/language.py'
        spec = importlib.util.spec_from_file_location('language', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for text in ('Hello', 'Привет, friend!', '', '123', 'Привіт', '你好'):
            self.assertEqual(module.russian_only(text), module.FALLBACK)
        self.assertEqual(module.russian_only(' Ёжик, привет! 42 '), 'Ёжик, привет! 42')


if __name__ == '__main__':
    unittest.main()
