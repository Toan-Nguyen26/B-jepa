"""Sequence feature extraction utilities."""
from __future__ import annotations


def compute_gc_content(sequence: str) -> float:
    """GC fraction for a DNA sequence."""
    if not sequence:
        return 0.0
    gc = sum(1 for c in sequence.upper() if c in "GC")
    return gc / len(sequence)


def compute_kmer_freq(sequence: str, k: int = 3) -> dict[str, float]:
    """k-mer frequency vector."""
    sequence = sequence.upper()
    counts: dict[str, int] = {}
    total = max(len(sequence) - k + 1, 1)
    for i in range(len(sequence) - k + 1):
        kmer = sequence[i : i + k]
        counts[kmer] = counts.get(kmer, 0) + 1
    return {kmer: count / total for kmer, count in counts.items()}


def compute_complexity(sequence: str) -> float:
    """Linguistic complexity score (unique k-mers / possible k-mers)."""
    k = 3
    sequence = sequence.upper()
    kmers = set()
    for i in range(len(sequence) - k + 1):
        kmers.add(sequence[i : i + k])
    possible = min(4**k, len(sequence) - k + 1)
    return len(kmers) / max(possible, 1)
