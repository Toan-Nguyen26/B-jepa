#!/usr/bin/env python3
"""Train BPE tokenizer on bacterial genome corpus.

Usage:
    python tools/train_tokenizer.py --data data/processed/pretrain_sequences.csv --output data/tokenizer/bpe_4096.json
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Train BPE tokenizer")
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--output", type=str, default="data/tokenizer/bpe_4096.json")
    parser.add_argument("--vocab-size", type=int, default=4096)
    parser.add_argument("--min-frequency", type=int, default=2)
    args = parser.parse_args()

    from tokenizers import Tokenizer, models, trainers, pre_tokenizers

    # Load sequences
    df = pd.read_csv(args.data)
    sequences = df["sequence"].tolist()

    # Configure BPE tokenizer
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    special_tokens = ["[PAD]", "[MASK]", "[CLS]", "[SEP]", "[UNK]"]
    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        special_tokens=special_tokens,
        show_progress=True,
    )

    # Train
    tokenizer.train_from_iterator(sequences, trainer=trainer)

    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    tokenizer.save(args.output)
    print(f"Tokenizer saved to {args.output} (vocab_size={tokenizer.get_vocab_size()})")


if __name__ == "__main__":
    main()
