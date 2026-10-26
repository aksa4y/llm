"""Entry point for: python -m guppylm"""

import os
import sys

CHECKPOINT_PATH = "checkpoints/best_model.pt"
TOKENIZER_PATH = "data/tokenizer.json"

def main():
    if len(sys.argv) < 2:
        print("GuppyLM — A tiny fish brain")
        print()
        print("Usage:")
        print("  python -m guppylm train        Train the model")
        print("  python -m guppylm prepare      Generate data & train tokenizer")
        print("  python -m guppylm chat         Chat with Guppy")
        print("  python -m guppylm download     Unavailable: original weights are English")
        return

    cmd = sys.argv[1]
    sys.argv = sys.argv[1:]

    if cmd == "prepare":
        import argparse
        from .prepare_data import prepare
        parser = argparse.ArgumentParser()
        parser.add_argument("--data-dir", default="data")
        prepare(parser.parse_args(sys.argv[1:]).data_dir)

    elif cmd == "train":
        from .train import main as train_main
        train_main()

    elif cmd == "download":
        raise SystemExit("Оригинальные веса англоязычные. Для русской версии выполните prepare и train.")

    elif cmd == "chat":
        from .inference import main as inference_main
        inference_main()

    else:
        print(f"Unknown command: {cmd}")
        print("Run 'python -m guppylm' for usage.")


main()
