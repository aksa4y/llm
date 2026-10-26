"""Prepare training data for GuppyLM."""

import json
import os
import random

random.seed(42)

DATA_DIR = "data"
VOCAB_SIZE = 4096

SPECIAL_TOKENS = [
    "<pad>",         # 0
    "<|im_start|>",  # 1
    "<|im_end|>",    # 2
]


def train_tokenizer(texts, save_path, vocab_size=VOCAB_SIZE):
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors

    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
        min_frequency=2,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )

    print(f"Training BPE tokenizer (vocab_size={vocab_size}) on {len(texts)} texts...")
    tokenizer.train_from_iterator(texts, trainer)
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    tokenizer.save(save_path)
    print(f"Tokenizer saved to {save_path} ({tokenizer.get_vocab_size()} tokens)")
    return tokenizer


def prepare(data_dir=DATA_DIR):
    from .russian_data import build
    build(data_dir)
    # Validation text must never participate in tokenizer training.
    with open(os.path.join(data_dir, "train.jsonl"), encoding="utf-8") as f:
        texts = [json.loads(line)["text"] for line in f]
    train_tokenizer(texts, os.path.join(data_dir, "tokenizer.json"))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=DATA_DIR)
    prepare(parser.parse_args().data_dir)
