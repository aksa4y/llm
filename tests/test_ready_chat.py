import unittest
from unittest.mock import Mock
from russian_chat import RussianChat, FALLBACK, is_russian, make_messages


class ReadyChatTests(unittest.TestCase):
    def test_language(self):
        for text in ('Привет! Как дела?', 'Используйте Python.', 'Ответ: 42.', '42'):
            self.assertTrue(is_russian(text), text)
        for text in ('Hello, how are you?', '你好', 'Привет 你好你好你好你好', 'Вот ответ: 小程序.', ''):
            self.assertFalse(is_russian(text), text)
        self.assertTrue(is_russian('Вот пример:\n```python\nprint("Hello")\n```'))

    def test_retry_and_fallback(self):
        engine = RussianChat.__new__(RussianChat)
        engine._generate = Mock(side_effect=['Hello', 'Привет!'])
        self.assertEqual(engine.answer('Hi'), 'Привет!')
        self.assertEqual(engine._generate.call_count, 2)
        engine._generate = Mock(return_value='Hello')
        self.assertEqual(engine.answer('Hi'), FALLBACK)

    def test_history_and_system(self):
        history = [{'role': 'user', 'content': 'Привет'}, {'role': 'assistant', 'content': 'Здравствуйте'}]
        messages = make_messages(history, 'Как дела?')
        self.assertEqual(messages[0]['role'], 'system')
        self.assertEqual(messages[1:3], history)
        self.assertEqual(len(history), 2)


class ContextTests(unittest.TestCase):
    def test_trim_old_turns_keep_system_and_question(self):
        engine = RussianChat.__new__(RussianChat)
        engine.count_tokens = Mock(side_effect=[8192, 300])
        engine._generate = Mock(return_value='Привет!')
        history = [{'role': 'user', 'content': 'старый вопрос'},
                   {'role': 'assistant', 'content': 'старый ответ'},
                   {'role': 'user', 'content': 'новый вопрос'},
                   {'role': 'assistant', 'content': 'новый ответ'}]
        engine.answer('текущий вопрос', history)
        messages = engine._generate.call_args.args[0]
        self.assertEqual(messages[0]['role'], 'system')
        self.assertEqual(messages[1:3], history[2:])
        self.assertEqual(messages[-1]['content'], 'текущий вопрос')
        self.assertEqual(len(history), 4)

    def test_chatml_and_generation_budget(self):
        engine = RussianChat.__new__(RussianChat)
        engine.model = Mock()
        engine.model.tokenize.return_value = [1, 2, 3]
        engine.model.create_completion.return_value = {'choices': [{'text': ' Привет! '}]}
        messages = make_messages([], 'Привет')
        self.assertTrue(engine.format_prompt(messages).endswith('<|im_start|>assistant\n'))
        self.assertEqual(engine._generate(messages, 32), 'Привет!')
        self.assertEqual(engine.model.create_completion.call_args.kwargs['prompt'], [1, 2, 3])
        engine.model.tokenize.return_value = [1] * 8190
        with self.assertRaises(ValueError):
            engine._generate(messages, 32)
