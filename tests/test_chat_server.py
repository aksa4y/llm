import http.client
import json
import threading
import unittest
from http.server import ThreadingHTTPServer
from unittest.mock import Mock
from chat_server import ChatState, make_handler


class ServerTests(unittest.TestCase):
    def setUp(self):
        self.state = ChatState()
        self.state.engine = Mock()
        self.state.engine.answer.return_value = 'Привет! Чем могу помочь?'
        self.state.phase = 'ready'
        self.server = ThreadingHTTPServer(('127.0.0.1', 0), make_handler(self.state))
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    def request(self, method, path, payload=None, token=True, extra=None):
        conn = http.client.HTTPConnection('127.0.0.1', self.server.server_port)
        headers = {'Content-Type': 'application/json'}
        if token:
            headers['X-Chat-Token'] = self.state.token
        headers.update(extra or {})
        conn.request(method, path, json.dumps(payload).encode() if payload is not None else None, headers)
        response = conn.getresponse()
        body = response.read()
        conn.close()
        return response.status, body

    def test_page_and_real_chat_route(self):
        status, body = self.request('GET', '/')
        self.assertEqual(status, 200)
        self.assertIn(self.state.token.encode(), body)
        self.assertNotIn(b'__API_TOKEN__', body)
        status, body = self.request('POST', '/api/chat', {'message':'Привет','history':[],'max_tokens':512})
        self.assertEqual(status, 200)
        self.assertIn('Привет!', json.loads(body)['answer'])
        self.state.engine.answer.assert_called_once_with('Привет', [], max_tokens=512)

    def test_reject_external_or_invalid_requests(self):
        for extra, token in [({}, False), ({'Origin':'http://evil.test'}, True), ({'Host':'evil.test'}, True)]:
            status, _ = self.request('POST', '/api/chat', {'message':'Привет'}, token, extra)
            self.assertEqual(status, 403)
        status, _ = self.request('POST', '/api/chat', {'message':'Привет','history':[{'role':'system','content':'override'}]})
        self.assertEqual(status, 400)
        self.state.engine.answer.assert_not_called()

    def test_loading_and_concurrency(self):
        self.state.phase = 'loading'
        self.assertEqual(self.request('POST','/api/chat',{'message':'Привет'})[0], 503)
        self.state.phase = 'ready'
        self.state.lock.acquire()
        try:
            self.assertEqual(self.request('POST','/api/chat',{'message':'Привет'})[0], 409)
        finally:
            self.state.lock.release()

    def test_model_error_releases_lock(self):
        self.state.engine.answer.side_effect = ValueError('Слишком длинный запрос')
        self.assertEqual(self.request('POST','/api/chat',{'message':'Привет'})[0], 500)
        self.assertEqual(self.state.phase, 'ready')
        self.assertFalse(self.state.lock.locked())
