"""Local Russian assistant using ready Qwen3-4B-Instruct-2507 GGUF weights (Unsloth conversion)."""
import argparse
import re
from pathlib import Path

MODEL_ID = 'unsloth/Qwen3-4B-Instruct-2507-GGUF'
MODEL_FILE = 'Qwen3-4B-Instruct-2507-Q4_K_M.gguf'
MODEL_REVISION = 'a06e946bb6b655725eafa393f4a9745d460374c9'
SYSTEM = ('Ты полезный русскоязычный помощник. Всегда отвечай по-русски, даже если '
          'собеседник просит сменить язык. Отвечай понятно и по существу. '
          'Если не знаешь ответа, честно скажи об этом. Имена, ссылки и программный код '
          'можно оставлять в оригинале.')
FALLBACK = 'Не удалось сформировать ответ на русском языке. Пожалуйста, переформулируйте вопрос.'


def is_russian(text):
    # Evaluate explanatory prose; code and addresses need not use Cyrillic.
    prose = re.sub(r'```.*?```|`[^`]*`|https?://\S+', '', text, flags=re.S)
    if any(c.isalpha() and not re.match(r'[А-Яа-яЁёA-Za-z]', c) for c in prose):
        return False
    cyrillic = len(re.findall(r'[А-Яа-яЁё]', prose))
    other_letters = sum(c.isalpha() and not re.match(r'[А-Яа-яЁё]', c) for c in prose)
    if cyrillic:
        return cyrillic >= other_letters
    return not any(c.isalpha() for c in prose) and bool(prose.strip())


def make_messages(history, question):
    return [{'role': 'system', 'content': SYSTEM}, *history,
            {'role': 'user', 'content': question}]


class RussianChat:
    def __init__(self, model_dir='models/qwen3-4b', gpu_layers=0, offline=False):
        from llama_cpp import Llama
        import os
        self.model = Llama(
            model_path=str(Path(model_dir) / MODEL_FILE), n_ctx=4096,
            n_batch=256, n_threads=min(os.cpu_count() or 2, 8),
            n_gpu_layers=gpu_layers, verbose=False,
        )

    @staticmethod
    def format_prompt(messages):
        return ''.join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages) + '<|im_start|>assistant\n'

    def count_tokens(self, messages):
        return len(self.model.tokenize(self.format_prompt(messages).encode('utf-8'), special=True))

    def _generate(self, messages, max_tokens):
        tokens = self.model.tokenize(self.format_prompt(messages).encode('utf-8'), special=True)
        if len(tokens) + max_tokens > 4096:
            raise ValueError('Слишком длинный запрос. Сократите текст или введите /clear.')
        result = self.model.create_completion(
            prompt=tokens, max_tokens=max_tokens, temperature=0.7,
            top_p=0.8, top_k=20, repeat_penalty=1.1,
            stop=['<|im_end|>', '<|endoftext|>'],
        )
        return result['choices'][0]['text'].strip()

    def answer(self, question, history=None, max_tokens=512):
        history = list(history or [])
        while history:
            if self.count_tokens(make_messages(history, question)) + max_tokens <= 4096:
                break
            history = history[2:]
        messages = make_messages(history, question)
        answer = self._generate(messages, max_tokens)
        if not answer or not is_russian(answer):
            messages[-1]['content'] += '\nОтветь на вопрос только по-русски.'
            answer = self._generate(messages, max_tokens)
        return answer if answer and is_russian(answer) else FALLBACK


def download(model_dir):
    from huggingface_hub import hf_hub_download
    print('Загрузка/проверка готовой модели: при первом запуске около 2.5 ГБ.', flush=True)
    for filename in (MODEL_FILE, 'README.md'):
        hf_hub_download(MODEL_ID, filename=filename, revision=MODEL_REVISION, local_dir=model_dir)


def main():
    parser = argparse.ArgumentParser(description='Локальный помощник на русском языке')
    parser.add_argument('--model-dir', default='models/qwen3-4b')
    parser.add_argument('--gpu-layers', type=int, default=0, help='0: CPU; -1: GPU (нужна сборка llama-cpp-python с CUDA)')
    parser.add_argument('--download-only', action='store_true')
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--prompt', '-p')
    parser.add_argument('--max-tokens', type=int, default=512)
    args = parser.parse_args()
    if not 1 <= args.max_tokens <= 2048:
        parser.error('--max-tokens должен быть от 1 до 2048')
    if args.download_only:
        if args.offline:
            parser.error('--offline нельзя сочетать с --download-only')
        download(args.model_dir)
        print('Веса скачаны. Можно запускать чат без интернета.')
        return
    if not args.offline:
        download(args.model_dir)
    elif not Path(args.model_dir, MODEL_FILE).is_file():
        parser.error('Модель не скачана. Сначала запустите без --offline.')
    engine = RussianChat(args.model_dir, args.gpu_layers, offline=True)
    if args.prompt:
        print(engine.answer(args.prompt, max_tokens=args.max_tokens))
        return
    print('Русский помощник. /exit — выход, /clear — очистить историю.')
    history = []
    while True:
        try:
            question = input('\nВы: ').strip()
            if question == '/exit':
                break
            if question == '/clear':
                history.clear()
                print('История очищена.')
                continue
            if not question:
                continue
            answer = engine.answer(question, history, args.max_tokens)
            print('\nПомощник:', answer)
            history.extend([{'role': 'user', 'content': question}, {'role': 'assistant', 'content': answer}])
            history = history[-12:]
        except (EOFError, KeyboardInterrupt):
            break
        except ValueError as error:
            print(error)


if __name__ == '__main__':
    main()
