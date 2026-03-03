"""DNA tokenizers: character-level (v3.1) and BPE (v4.0)."""
from __future__ import annotations

from typing import Optional

import torch


class CharTokenizer:
    """Character-level A/C/G/T/N tokenizer with special tokens."""

    SPECIAL_TOKENS = {"[PAD]": 0, "[MASK]": 1, "[CLS]": 2, "[SEP]": 3, "[UNK]": 4}
    NUCLEOTIDES = {"A": 5, "C": 6, "G": 7, "T": 8, "N": 9}

    def __init__(self):
        self.token_to_id = {**self.SPECIAL_TOKENS, **self.NUCLEOTIDES}
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}
        self.pad_id = 0
        self.mask_id = 1
        self.cls_id = 2
        self.sep_id = 3
        self.unk_id = 4

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    def encode(self, sequence: str, add_special_tokens: bool = False) -> list[int]:
        ids = [self.token_to_id.get(c.upper(), self.unk_id) for c in sequence]
        if add_special_tokens:
            ids = [self.cls_id] + ids + [self.sep_id]
        return ids

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        tokens = []
        for i in ids:
            tok = self.id_to_token.get(i, "")
            if skip_special and i < 5:
                continue
            tokens.append(tok)
        return "".join(tokens)

    def batch_encode(
        self,
        sequences: list[str],
        max_length: Optional[int] = None,
        add_special_tokens: bool = True,
    ) -> dict[str, torch.Tensor]:
        encoded = [self.encode(s, add_special_tokens=add_special_tokens) for s in sequences]
        if max_length is None:
            max_length = max(len(e) for e in encoded)

        input_ids = torch.full((len(sequences), max_length), self.pad_id, dtype=torch.long)
        attention_mask = torch.zeros(len(sequences), max_length, dtype=torch.bool)

        for i, ids in enumerate(encoded):
            length = min(len(ids), max_length)
            input_ids[i, :length] = torch.tensor(ids[:length], dtype=torch.long)
            attention_mask[i, :length] = True

        return {"input_ids": input_ids, "attention_mask": attention_mask}


class BPETokenizer:
    """BPE tokenizer wrapping HuggingFace tokenizers library."""

    def __init__(self, tokenizer_path: str):
        from tokenizers import Tokenizer
        self._tokenizer = Tokenizer.from_file(tokenizer_path)
        self.pad_id = self._tokenizer.token_to_id("[PAD]") or 0
        self.mask_id = self._tokenizer.token_to_id("[MASK]") or 1
        self.cls_id = self._tokenizer.token_to_id("[CLS]") or 2
        self.sep_id = self._tokenizer.token_to_id("[SEP]") or 3
        self.unk_id = self._tokenizer.token_to_id("[UNK]") or 4

    @property
    def vocab_size(self) -> int:
        return self._tokenizer.get_vocab_size()

    def encode(self, sequence: str, add_special_tokens: bool = False) -> list[int]:
        encoding = self._tokenizer.encode(sequence, add_special_tokens=False)
        ids = encoding.ids
        if add_special_tokens:
            ids = [self.cls_id] + ids + [self.sep_id]
        return ids

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        return self._tokenizer.decode(ids, skip_special_tokens=skip_special)

    def batch_encode(
        self,
        sequences: list[str],
        max_length: Optional[int] = None,
        add_special_tokens: bool = True,
    ) -> dict[str, torch.Tensor]:
        encoded = [self.encode(s, add_special_tokens=add_special_tokens) for s in sequences]
        if max_length is None:
            max_length = max(len(e) for e in encoded)

        input_ids = torch.full((len(sequences), max_length), self.pad_id, dtype=torch.long)
        attention_mask = torch.zeros(len(sequences), max_length, dtype=torch.bool)

        for i, ids in enumerate(encoded):
            length = min(len(ids), max_length)
            input_ids[i, :length] = torch.tensor(ids[:length], dtype=torch.long)
            attention_mask[i, :length] = True

        return {"input_ids": input_ids, "attention_mask": attention_mask}


def get_tokenizer(version: str = "v4.0", tokenizer_path: Optional[str] = None):
    """Factory: picks tokenizer based on version string."""
    if version == "v3.1" or tokenizer_path is None:
        return CharTokenizer()
    return BPETokenizer(tokenizer_path)
