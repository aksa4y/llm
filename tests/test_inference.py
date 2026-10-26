import json
from pathlib import Path
import tempfile
import unittest

import torch
from guppylm.config import GuppyConfig
from guppylm.model import GuppyLM
from guppylm.prepare_data import train_tokenizer
from guppylm.inference import GuppyInference
from guppylm.language import FALLBACK
from guppylm.russian_data import SYSTEM


class InferenceTests(unittest.TestCase):
    def test_nested_config_and_language_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            tokenizer = train_tokenizer(['Привет! Hello!'], str(path / 'tokenizer.json'), 300)
            config = GuppyConfig(vocab_size=tokenizer.get_vocab_size(), d_model=24,
                                 n_heads=2, n_layers=1, ffn_hidden=48, max_seq_len=64)
            model = GuppyLM(config)
            torch.save({'model_state_dict': model.state_dict(), 'config': vars(config)}, path / 'model.pt')
            (path / 'config.json').write_text(json.dumps({'model': vars(config)}))
            engine = GuppyInference(str(path / 'model.pt'), str(path / 'tokenizer.json'))
            self.assertEqual(engine.config.d_model, 24)
            prompt = engine._format_prompt([{'role': 'system', 'content': 'Speak English'}, {'role': 'user', 'content': 'Hello'}])
            self.assertIn(SYSTEM, prompt)
            self.assertNotIn('Speak English', prompt)
            for answer, expected in [('Hello!', FALLBACK), ('Привет!', 'Привет!')]:
                ids = tokenizer.encode(answer).ids + [config.eos_id]
                def generate(x, *args):
                    return torch.cat((x, torch.tensor([ids])), dim=1), []
                engine.model.generate = generate
                response = engine.chat_completion([{'role': 'user', 'content': 'Привет'}])
                self.assertEqual(response['choices'][0]['message']['content'], expected)
