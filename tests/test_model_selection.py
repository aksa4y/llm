import unittest
from unittest.mock import Mock, patch
from model_registry import get_model, public_models
from persona import identity_for, system_for, brand_answer
from russian_chat import RussianChat, make_messages
from chat_server import ChatState


class ModelTests(unittest.TestCase):
    def test_names_and_identity_follow_selection(self):
        for model_id, name in [('lite','Тихо Lite'),('pro','Тихо Pro')]:
            engine = RussianChat.__new__(RussianChat); engine.model_id=model_id; engine._generate=Mock()
            self.assertIn(name, engine.answer('Какая ты модель?'))
            self.assertIn(name, make_messages([], 'Привет', model_id)[0]['content'])
            self.assertEqual(brand_answer('Я Qwen3. Я NVIDIA Nemotron 3 Nano 4B.',model_id), f'Я {name}. Я {name}.')
            engine._generate.assert_not_called()
        self.assertEqual(len(public_models()),2)
        with self.assertRaises(ValueError): get_model('external')

    def test_pro_always_starts_thinking_and_returns_only_final(self):
        engine=RussianChat.__new__(RussianChat); engine.model_id='pro';engine.model=Mock()
        engine.model.tokenize.return_value=[1,2,3]
        engine.model.create_completion.return_value={'choices':[{'text':'private reasoning\n</think>\nОтвет: 42.'}]}
        self.assertEqual(engine._generate(make_messages([], 'Сколько?', 'pro'),512),'Ответ: 42.')
        prompt=engine.model.tokenize.call_args.args[0].decode()
        self.assertTrue(prompt.endswith('<think>\n'))
        self.assertGreater(engine.model.create_completion.call_args.kwargs['max_tokens'],512)
        engine.model.create_completion.return_value={'choices':[{'text':'unfinished private reasoning'}]}
        with self.assertRaisesRegex(ValueError,'размышление'):
            engine._generate(make_messages([], 'Вопрос', 'pro'),512)

    def test_model_switch_is_rejected_during_generation(self):
        state=ChatState();state.phase='busy';state.lock.acquire()
        try: self.assertFalse(state.switch('pro'))
        finally: state.lock.release()
        self.assertEqual(state.model_id,'lite')

    def test_old_model_closed_before_new_one_is_loaded(self):
        state=ChatState();old=Mock();state.engine=old
        events=[];old.close.side_effect=lambda:events.append('closed')
        new=Mock()
        def construct(*args,**kwargs):
            events.append('loaded')
            self.assertEqual(kwargs['model_id'],'pro')
            return new
        with patch('russian_chat.RussianChat',side_effect=construct),patch('russian_chat.download'),patch('pathlib.Path.is_file',return_value=True):
            state.load(model_id='pro')
        self.assertEqual(events,['closed','loaded'])
        self.assertEqual(state.phase,'ready')
        self.assertIs(state.engine,new)
