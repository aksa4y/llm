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
        engine.tokenizer = Mock()
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
