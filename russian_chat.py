"""Local Russian assistant using ready Qwen3-4B-Instruct-2507 GGUF weights (Unsloth conversion)."""
import argparse
import re
from pathlib import Path

MODEL_ID = 'unsloth/Qwen3-4B-Instruct-2507-GGUF'
MODEL_FILE = 'Qwen3-4B-Instruct-2507-Q4_K_M.gguf'
MODEL_REVISION = 'a06e946bb6b655725eafa393f4a9745d460374c9'
from persona import SYSTEM, IDENTITY, identity_question, brand_answer, identity_for, system_for
from model_registry import get_model
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


def make_messages(history, question, model_id='lite'):
    return [{'role': 'system', 'content': system_for(model_id)}, *history,
            {'role': 'user', 'content': question}]


class RussianChat:
    def __init__(self, model_dir=None, gpu_layers=0, offline=False, model_id='lite'):
        from llama_cpp import Llama
        import os
        self.model_id = model_id
        spec = get_model(model_id)
        model_dir = model_dir or spec.directory
        self.model = Llama(
            model_path=str(Path(model_dir) / spec.filename), n_ctx=8192,
            n_batch=256, n_threads=min(os.cpu_count() or 2, 8),
            n_gpu_layers=gpu_layers, verbose=False,
        )

    def close(self):
        self.model.close()

    @staticmethod
    def format_prompt(messages, reasoning=False):
        if reasoning:
            messages = [{**m, 'content': '<think></think>' + m['content']} if m['role'] == 'assistant' else m for m in messages]
        return ''.join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages) + '<|im_start|>assistant\n' + ('<think>\n' if reasoning else '')

    def count_tokens(self, messages):
        return len(self.model.tokenize(self.format_prompt(messages, get_model(getattr(self, 'model_id', 'lite')).reasoning).encode('utf-8'), special=True))

    def _generate(self, messages, max_tokens):
        tokens = self.model.tokenize(self.format_prompt(messages, get_model(getattr(self, 'model_id', 'lite')).reasoning).encode('utf-8'), special=True)
        if len(tokens) + max_tokens > 8192:
            raise ValueError('Слишком длинный запрос. Сократите текст или введите /clear.')
        reasoning = get_model(getattr(self, 'model_id', 'lite')).reasoning
        generation_budget = min(8192 - len(tokens), max_tokens + (3072 if reasoning else 0))
        result = self.model.create_completion(
            prompt=tokens, max_tokens=generation_budget, temperature=0.6 if reasoning else 0.7,
            top_p=0.8, top_k=20, repeat_penalty=1.1,
            stop=['<|im_end|>', '<|endoftext|>'],
        )
        text = result['choices'][0]['text'].strip()
        if reasoning:
            if '</think>' not in text:
                raise ValueError('Тихо Pro не успел завершить размышление в доступном контексте. Сократи вопрос или вложения и попробуй снова.')
            text = text.rsplit('</think>', 1)[1].strip()
            if not text:
                raise ValueError('Тихо Pro завершил размышление, но лимита не хватило на итоговый ответ. Сократи запрос.')
        return text

    def answer(self, question, history=None, max_tokens=1536):
        if identity_question(question):
            return identity_for(getattr(self, 'model_id', 'lite'))
        history = list(history or [])
        while history:
            if self.count_tokens(make_messages(history, question, getattr(self, 'model_id', 'lite'))) + max_tokens + (2048 if get_model(getattr(self, 'model_id', 'lite')).reasoning else 0) <= 8192:
                break
            history = history[2:]
        messages = make_messages(history, question, getattr(self, 'model_id', 'lite'))
        answer = self._generate(messages, max_tokens)
        if not answer or not is_russian(answer):
            messages[-1]['content'] += '\nОтветь на вопрос только по-русски.'
            answer = self._generate(messages, max_tokens)
        return brand_answer(answer, getattr(self, 'model_id', 'lite')) if answer and is_russian(answer) else FALLBACK


def download(model_dir=None, model_id='lite'):
    spec = get_model(model_id)
    model_dir = model_dir or spec.directory
    from huggingface_hub import hf_hub_download
    print(f'Загрузка/проверка {spec.name}: около {"2.9" if spec.reasoning else "2.5"} ГБ при первом запуске.', flush=True)
    for filename in (spec.filename, 'README.md') + (('LICENSE',) if spec.id == 'pro' else ()):
        hf_hub_download(spec.repository, filename=filename, revision=spec.revision, local_dir=model_dir)


def main():
    parser = argparse.ArgumentParser(description='Локальный помощник на русском языке')
    parser.add_argument('--model', choices=['lite', 'pro'], default='lite')
    parser.add_argument('--model-dir', default=None)
    parser.add_argument('--gpu-layers', type=int, default=0, help='0: CPU; -1: GPU (нужна сборка llama-cpp-python с CUDA)')
    parser.add_argument('--download-only', action='store_true')
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--prompt', '-p')
    parser.add_argument('--max-tokens', type=int, default=1536)
    args = parser.parse_args()
    if not 1 <= args.max_tokens <= 3072:
        parser.error('--max-tokens должен быть от 1 до 3072')
    args.model_dir = args.model_dir or get_model(args.model).directory
    if args.download_only:
        if args.offline:
            parser.error('--offline нельзя сочетать с --download-only')
        download(args.model_dir, args.model)
        print('Веса скачаны. Можно запускать чат без интернета.')
        return
    if not args.offline:
        download(args.model_dir, args.model)
    elif not Path(args.model_dir, get_model(args.model).filename).is_file():
        parser.error('Модель не скачана. Сначала запустите без --offline.')
    engine = RussianChat(args.model_dir, args.gpu_layers, offline=True, model_id=args.model)
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
