"""Datasets for B-JEPA pretraining."""
from __future__ import annotations

from typing import Optional

import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler

from bdna_jepa.data.tokenizer import CharTokenizer, BPETokenizer


class BacterialGenomeDataset(Dataset):
    """Standard fragment dataset from CSV.

    Expected columns: sequence, genome (optional), species (optional), gc_content (optional)
    """

    def __init__(
        self,
        csv_path: str,
        tokenizer: CharTokenizer | BPETokenizer,
        max_length: int = 1024,
        add_special_tokens: bool = False,
    ):
        self.df = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.add_special_tokens = add_special_tokens

        # Build genome -> indices mapping for fragment sampling
        self.genome_to_indices: dict[str, list[int]] = {}
        if "genome" in self.df.columns:
            for idx, genome in enumerate(self.df["genome"]):
                self.genome_to_indices.setdefault(str(genome), []).append(idx)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        sequence = str(row["sequence"])

        ids = self.tokenizer.encode(sequence, add_special_tokens=self.add_special_tokens)
        ids = ids[: self.max_length]

        tokens = torch.tensor(ids, dtype=torch.long)

        item = {"tokens": tokens, "idx": idx}

        if "gc_content" in self.df.columns:
            item["gc_content"] = float(row["gc_content"])
        if "species" in self.df.columns:
            item["species"] = str(row["species"])
        if "genome" in self.df.columns:
            item["genome"] = str(row["genome"])

        return item


def collate_fn(batch: list[dict]) -> dict:
    """Collate with dynamic padding."""
    max_len = max(item["tokens"].size(0) for item in batch)
    pad_id = 0

    tokens = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros(len(batch), max_len, dtype=torch.bool)
    indices = []
    gc_values = []

    for i, item in enumerate(batch):
        length = item["tokens"].size(0)
        tokens[i, :length] = item["tokens"]
        attention_mask[i, :length] = True
        indices.append(item["idx"])
        if "gc_content" in item:
            gc_values.append(item["gc_content"])

    result = {
        "tokens": tokens,
        "attention_mask": attention_mask,
        "indices": torch.tensor(indices, dtype=torch.long),
    }
    if gc_values:
        result["gc_content"] = torch.tensor(gc_values, dtype=torch.float)
    return result


class GenomeAwareBatchSampler(Sampler):
    """Samples K fragments per genome per batch for fragment JEPA."""

    def __init__(
        self,
        genome_to_indices: dict[str, list[int]],
        fragments_per_genome: int = 4,
        batch_size: int = 256,
        drop_last: bool = True,
        shuffle: bool = True,
    ):
        self.fragments_per_genome = fragments_per_genome
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle

        # Filter genomes with enough fragments
        self.valid_genomes = {
            g: idxs for g, idxs in genome_to_indices.items()
            if len(idxs) >= fragments_per_genome
        }
        self.genome_list = list(self.valid_genomes.keys())
        self.genomes_per_batch = batch_size // fragments_per_genome

    def __iter__(self):
        import random as rng

        genome_order = list(self.genome_list)
        if self.shuffle:
            rng.shuffle(genome_order)

        batch = []
        for genome in genome_order:
            indices = list(self.valid_genomes[genome])
            if self.shuffle:
                rng.shuffle(indices)
            selected = indices[: self.fragments_per_genome]
            batch.extend(selected)

            if len(batch) >= self.batch_size:
                yield batch[: self.batch_size]
                batch = batch[self.batch_size :]

        if batch and not self.drop_last:
            yield batch

    def __len__(self) -> int:
        n_batches = len(self.genome_list) // self.genomes_per_batch
        if not self.drop_last and len(self.genome_list) % self.genomes_per_batch:
            n_batches += 1
        return n_batches
