"""Loopback-only browser interface for the local Qwen assistant."""
import argparse
import json
from pathlib import Path
import secrets
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import webbrowser
from model_registry import get_model, public_models

ROOT = Path(__file__).resolve().parent


class ChatState:
    def __init__(self):
        self.token = secrets.token_urlsafe(32)
        self.phase = 'loading'
        self.detail = 'Подготовка модели'
        self.engine = None
        self.lock = threading.Lock()
        self.model_id = 'lite'
        self.lite_dir = None
        self.model_dirs = {}
        self.gpu_layers = 0
        self.offline = False

    def load(self, model_dir=None, gpu_layers=0, offline=False, model_id='lite'):
        try:
            from russian_chat import RussianChat, download
            spec = get_model(model_id)
            self.model_id = model_id
            path = Path(model_dir) if model_dir else ROOT / spec.directory
            previous, self.engine = self.engine, None
            if previous is not None:
                previous.close()
                del previous
                __import__('gc').collect()
            if not (path / spec.filename).is_file():
                if offline:
                    raise ValueError('Веса не найдены. Запусти без --offline, чтобы скачать выбранную модель.')
                self.detail = f'Скачивание {spec.name}: около {"2,9" if model_id == "pro" else "2,5"} ГБ. Это нужно только один раз.'
                download(str(path), model_id)
            self.detail = f'Загрузка {spec.name} в память'
            self.engine = RussianChat(str(path), gpu_layers, offline=True, model_id=model_id)
            self.phase, self.detail = 'ready', f'{spec.name} готов к разговору'
        except Exception as error:
            self.phase, self.detail = 'error', str(error)

    def switch(self, model_id):
        spec = get_model(model_id)
        if not self.lock.acquire(blocking=False):
            return False
        if model_id == self.model_id and self.phase == 'ready':
            self.lock.release()
            return True
        self.model_id = model_id
        self.phase, self.detail = 'loading', f'Подготовка {spec.name}'
        def worker():
            try:
                self.load(self.model_dirs.get(model_id, self.lite_dir if model_id == 'lite' else None), self.gpu_layers, self.offline, model_id)
            finally:
                self.lock.release()
        threading.Thread(target=worker, daemon=True).start()
        return True


def make_handler(state):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass  # Do not put private conversations in request logs.

        def valid_host(self):
            port = self.server.server_port
            return self.headers.get('Host') in (f'127.0.0.1:{port}', f'localhost:{port}')

        def respond(self, status, data, kind='application/json; charset=utf-8'):
            body = json.dumps(data, ensure_ascii=False).encode() if isinstance(data, dict) else data
            self.send_response(status)
            self.send_header('Content-Type', kind)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; object-src 'none'; base-uri 'none'; frame-ancestors 'none'")
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_GET(self):
            if not self.valid_host():
                return self.respond(403, {'error': 'Недопустимый адрес'})
            route = self.path.split('?')[0]
            if route == '/api/status':
                return self.respond(200, {'phase': state.phase, 'detail': state.detail, 'model_id': state.model_id, 'model_name': get_model(state.model_id).name, 'models': public_models()})
            files = {'/': ('index.html', 'text/html; charset=utf-8'),
                     '/style.css': ('style.css', 'text/css; charset=utf-8'),
                     '/app.js': ('app.js', 'text/javascript; charset=utf-8')}
            if route not in files:
                return self.respond(404, {'error': 'Не найдено'})
            filename, kind = files[route]
            data = (ROOT / 'ui' / filename).read_bytes()
            if filename == 'index.html':
                data = data.replace(b'__API_TOKEN__', state.token.encode())
            self.respond(200, data, kind)

        def do_POST(self):
            if not self.valid_host() or self.headers.get('X-Chat-Token') != state.token:
                return self.respond(403, {'error': 'Обновите страницу и повторите запрос'})
            origin = self.headers.get('Origin')
            if origin and origin not in (f'http://127.0.0.1:{self.server.server_port}', f'http://localhost:{self.server.server_port}'):
                return self.respond(403, {'error': 'Недопустимый источник'})
            if self.path not in ('/api/chat', '/api/attachments', '/api/model'):
                return self.respond(404, {'error': 'Не найдено'})
            try:
                size = int(self.headers.get('Content-Length', '0'))
                if not 0 < size <= (30 * 1024 * 1024 if self.path == '/api/attachments' else 2 * 1024 * 1024):
                    return self.respond(413, {'error': 'Слишком большой запрос'})
                payload = json.loads(self.rfile.read(size))
                if self.path == '/api/model':
                    model_id = payload.get('model_id')
                    get_model(model_id)
                    if not state.switch(model_id):
                        return self.respond(409, {'error': 'Дождись окончания ответа или загрузки модели.'})
                    return self.respond(200, {'model_id': state.model_id, 'phase': state.phase})
                if self.path == '/api/attachments':
                    from attachments import unpack_files
                    try:
                        return self.respond(200, {'attachments': unpack_files(payload.get('files'))})
                    except Exception as error:
                        return self.respond(400, {'error': str(error)})
                question, history = payload.get('message'), payload.get('history', [])
                limit = payload.get('max_tokens', 1536)
                if payload.get('model_id', state.model_id) != state.model_id:
                    return self.respond(409, {'error': 'Модель изменилась. Обнови её выбор и повтори отправку.'})
                if not isinstance(question, str) or not question.strip() or len(question) > 16000:
                    raise ValueError('Введите сообщение длиной до 16 000 символов')
                if type(limit) is not int or limit not in (256, 512, 1024, 1536, 2048, 3072):
                    raise ValueError('Неверный лимит ответа')
                if not isinstance(history, list) or len(history) > 12:
                    raise ValueError('Слишком длинная история')
                if any(not isinstance(m, dict) or m.get('role') not in ('user', 'assistant') or not isinstance(m.get('content'), str) or len(m['content']) > 90000 for m in history):
                    raise ValueError('Неверный формат истории')
            except (ValueError, TypeError, AttributeError) as error:
                return self.respond(400, {'error': str(error)})
            if state.phase not in ('ready', 'busy'):
                return self.respond(503, {'error': state.detail})
            if not state.lock.acquire(blocking=False):
                return self.respond(409, {'error': 'Модель уже отвечает. Дождитесь завершения.'})
            try:
                state.phase = 'busy'
                state.detail = f'{get_model(state.model_id).name} размышляет…' if state.model_id == 'pro' else 'Тихо Lite готовит ответ…'
                from attachments import attach_context
                context, truncated = attach_context(question.strip(), payload.get('attachments', []), state.engine, limit)
                answer = state.engine.answer(context, history, max_tokens=limit)
                status, response = 200, {'answer': answer, 'context': context, 'truncated': truncated, 'model_id': state.model_id, 'model_name': get_model(state.model_id).name}
            except Exception as error:
                status, response = 500, {'error': str(error)}
            finally:
                state.phase = 'ready'
                state.detail = f'{get_model(state.model_id).name} готов к разговору'
                state.lock.release()
            self.respond(status, response)
    return Handler


def main():
    parser = argparse.ArgumentParser(description='Локальный чат в браузере')
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--model', choices=['lite', 'pro'], default='lite')
    parser.add_argument('--model-dir', default=None)
    parser.add_argument('--gpu-layers', type=int, default=0)
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--no-browser', action='store_true')
    args = parser.parse_args()
    state = ChatState()
    if args.model_dir:
        state.model_dirs[args.model] = args.model_dir
    state.gpu_layers = args.gpu_layers
    state.offline = args.offline
    try:
        server = ThreadingHTTPServer(('127.0.0.1', args.port), make_handler(state))
    except OSError as error:
        parser.exit(1, f'Не удалось открыть порт: {error}. Попробуйте --port 8766\n')
    url = f'http://127.0.0.1:{server.server_port}'
    print(f'Чат: {url}\nОставьте это окно открытым. Ctrl+C — завершить.', flush=True)
    state.switch(args.model)
    if not args.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
