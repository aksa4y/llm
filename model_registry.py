"""Explicit model sources and user-facing names; weights retain their source licenses."""
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    id: str
    name: str
    repository: str
    filename: str
    revision: str
    directory: str
    reasoning: bool = False


MODELS = {
    'lite': ModelSpec('lite', 'Тихо Lite', 'unsloth/Qwen3-4B-Instruct-2507-GGUF',
        'Qwen3-4B-Instruct-2507-Q4_K_M.gguf', 'a06e946bb6b655725eafa393f4a9745d460374c9', 'models/qwen3-4b'),
    'pro': ModelSpec('pro', 'Тихо Pro', 'nvidia/NVIDIA-Nemotron-3-Nano-4B-GGUF',
        'NVIDIA-Nemotron3-Nano-4B-Q4_K_M.gguf', 'ba223d14e45525f7fae81db77ea8cabeb2fc6c25', 'models/nemotron-3-nano-4b', True),
}


def get_model(model_id):
    if model_id not in MODELS:
        raise ValueError('Неизвестная модель. Выберите Тихо Lite или Тихо Pro.')
    return MODELS[model_id]


def public_models():
    return [{'id': m.id, 'name': m.name, 'reasoning': m.reasoning} for m in MODELS.values()]
