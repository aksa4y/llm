"""Local Russian assistant using official, already trained Qwen3 weights."""
import argparse
import re
from pathlib import Path

MODEL_ID = 'Qwen/Qwen3-1.7B'
MODEL_REVISION = '70d244cc86ccca08cf5af4e1e306ecf908b1ad5e'
SYSTEM = ('Ты полезный русскоязычный помощник. Всегда отвечай по-русски, даже если '
          'собеседник просит сменить язык. Отвечай понятно и по существу. '
          'Если не знаешь ответа, честно скажи об этом. Имена, ссылки и программный код '
          'можно оставлять в оригинале. Не выдавай себя за рыбку.')
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
    def __init__(self, model_dir='models/qwen3-1.7b', device='auto', offline=False, precision='auto'):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.device = 'cuda' if device == 'auto' and torch.cuda.is_available() else ('cpu' if device == 'auto' else device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=offline, trust_remote_code=False)
        dtype = {'float32': torch.float32, 'float16': torch.float16, 'bfloat16': torch.bfloat16}.get(precision)
        if dtype is None:
            dtype = torch.float16 if self.device == 'cuda' else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            model_dir, torch_dtype=dtype,
            local_files_only=offline, trust_remote_code=False, use_safetensors=True,
        ).to(self.device).eval()

    def _generate(self, messages, max_tokens):
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)
        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)
        # Reject an overlong current question rather than silently losing the system prompt.
        if inputs.input_ids.shape[1] + max_tokens > 4096:
            raise ValueError('Слишком длинный запрос. Сократите текст или введите /clear.')
        with self.torch.inference_mode():
            ids = self.model.generate(
                **inputs, max_new_tokens=max_tokens, do_sample=True, temperature=0.6,
                top_p=0.95, top_k=20, repetition_penalty=1.1,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        output = ids[0, inputs.input_ids.shape[1]:].tolist()
        end_thinking = self.tokenizer.convert_tokens_to_ids('</think>')
        if end_thinking in output:
            output = output[len(output) - output[::-1].index(end_thinking):]
        else:
            # Never show an unfinished reasoning block as the final answer.
            return ''
        return self.tokenizer.decode(output, skip_special_tokens=True).strip()

    def answer(self, question, history=None, max_tokens=768):
        history = list(history or [])
        while history:
            prompt = self.tokenizer.apply_chat_template(make_messages(history, question), tokenize=True,
                                                        add_generation_prompt=True, enable_thinking=True)
            if len(prompt) + max_tokens <= 4096:
                break
            history = history[2:]
        messages = make_messages(history, question)
        answer = self._generate(messages, max_tokens)
        if not answer or not is_russian(answer):
            messages[-1]['content'] += '\nОтветь на вопрос только по-русски.'
            answer = self._generate(messages, max_tokens)
        return answer if answer and is_russian(answer) else FALLBACK


def download(model_dir):
    from huggingface_hub import snapshot_download
    # A complete local snapshot includes weights, tokenizer, config and license.
    snapshot_download(MODEL_ID, revision=MODEL_REVISION, local_dir=model_dir,
                      allow_patterns=['*.json', '*.safetensors', '*.txt', 'LICENSE', 'README.md'])


def main():
    parser = argparse.ArgumentParser(description='Локальный помощник на русском языке')
    parser.add_argument('--model-dir', default='models/qwen3-1.7b')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--precision', choices=['auto', 'float32', 'float16', 'bfloat16'], default='auto')
    parser.add_argument('--download-only', action='store_true')
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--prompt', '-p')
    parser.add_argument('--max-tokens', type=int, default=768)
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
    elif not Path(args.model_dir, 'config.json').is_file():
        parser.error('Модель не скачана. Сначала запустите без --offline.')
    engine = RussianChat(args.model_dir, args.device, offline=True, precision=args.precision)
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
