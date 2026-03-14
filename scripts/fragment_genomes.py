#!/usr/bin/env python3
"""
Generate 10M+ fragments from bacterial genomes for B-JEPA v7 training.

Usage:
    python scripts/fragment_genomes.py \
        --input data/genomes/ \
        --output data/processed/pretrain_10M.csv \
        --window 2048 --stride 512 --max-fragments 10000000
"""

import argparse
import csv
import os
import random
from pathlib import Path


def gc_content(seq):
    """Compute GC fraction of a DNA sequence."""
    seq = seq.upper()
    gc = sum(1 for c in seq if c in 'GC')
    total = sum(1 for c in seq if c in 'ACGT')
    return gc / max(total, 1)


def fragment_genome(fasta_path, window=2048, stride=512, min_len=1024):
    """Fragment a single genome FASTA into overlapping windows."""
    genome_id = Path(fasta_path).stem  # e.g., GCF_045571495.1_LMG29515_v1_genomic

    # Read all contigs, concatenate with N spacers
    sequence = []
    with open(fasta_path) as f:
        for line in f:
            if line.startswith('>'):
                if sequence:
                    sequence.append('N' * 100)  # Spacer between contigs
            else:
                sequence.append(line.strip().upper())
    full_seq = ''.join(sequence)

    fragments = []
    for start in range(0, len(full_seq) - min_len + 1, stride):
        frag = full_seq[start:start + window]
        # Skip fragments with >10% N's (contig boundaries, low quality)
        n_count = frag.count('N')
        if n_count / len(frag) > 0.10:
            continue
        if len(frag) < min_len:
            continue
        gc = gc_content(frag)
        fragments.append((frag, genome_id, gc))

    return fragments


def main():
    parser = argparse.ArgumentParser(description="Fragment bacterial genomes for pretraining")
    parser.add_argument("--input", type=str, required=True, help="Directory with .fna genome files")
    parser.add_argument("--output", type=str, required=True, help="Output CSV path")
    parser.add_argument("--window", type=int, default=2048, help="Fragment window size (bp)")
    parser.add_argument("--stride", type=int, default=512, help="Stride between fragments (bp)")
    parser.add_argument("--min-len", type=int, default=1024, help="Minimum fragment length")
    parser.add_argument("--max-fragments", type=int, default=10_000_000, help="Max total fragments")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    # Find all genome files
    genome_dir = Path(args.input)
    fasta_files = sorted(genome_dir.glob("*.fna"))
    if not fasta_files:
        fasta_files = sorted(genome_dir.glob("**/*.fna"))
    print(f"Found {len(fasta_files)} genome files in {genome_dir}")

    # Fragment all genomes
    all_fragments = []
    for i, fasta in enumerate(fasta_files):
        frags = fragment_genome(fasta, args.window, args.stride, args.min_len)
        all_fragments.extend(frags)
        if (i + 1) % 500 == 0:
            print(f"  Processed {i+1}/{len(fasta_files)} genomes, {len(all_fragments):,} fragments so far")

    print(f"\nTotal fragments before sampling: {len(all_fragments):,}")

    # Subsample if needed
    if len(all_fragments) > args.max_fragments:
        random.shuffle(all_fragments)
        all_fragments = all_fragments[:args.max_fragments]
        print(f"Subsampled to {args.max_fragments:,} fragments")
    else:
        random.shuffle(all_fragments)

    # Write CSV
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['sequence', 'genome', 'gc_content'])
        for seq, genome, gc in all_fragments:
            writer.writerow([seq, genome, f"{gc:.4f}"])

    # Stats
    genomes = set(g for _, g, _ in all_fragments)
    gcs = [gc for _, _, gc in all_fragments]
    print(f"\nOutput: {output_path}")
    print(f"  Fragments: {len(all_fragments):,}")
    print(f"  Genomes:   {len(genomes):,}")
    print(f"  GC range:  {min(gcs):.3f} - {max(gcs):.3f}")
    print(f"  GC mean:   {sum(gcs)/len(gcs):.3f}")


if __name__ == "__main__":
    main()
