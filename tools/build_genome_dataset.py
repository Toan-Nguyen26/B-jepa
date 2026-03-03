#!/usr/bin/env python3
"""Build pretraining dataset: download genomes + extract fragments + compute features.

Usage:
    python tools/build_genome_dataset.py --output data/processed/ --n-genomes 800
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bdna_jepa.utils.logging import get_logger

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Build genome dataset")
    parser.add_argument("--output", type=str, default="data/processed/")
    parser.add_argument("--genomes-dir", type=str, default="data/genomes/")
    parser.add_argument("--n-genomes", type=int, default=800)
    parser.add_argument("--window", type=int, default=2048)
    parser.add_argument("--stride", type=int, default=512)
    args = parser.parse_args()

    # Step 1: Download genomes from NCBI
    logger.info(f"Downloading {args.n_genomes} bacterial genomes...")
    # This would use ncbi-datasets-cli or Biopython Entrez
    # For now, expects pre-downloaded FASTA files in genomes_dir

    # Step 2: Extract windows
    from bdna_jepa.utils.features import compute_gc_content
    import pandas as pd
    from pathlib import Path
    import glob

    fasta_files = glob.glob(os.path.join(args.genomes_dir, "*.fna")) + \
                  glob.glob(os.path.join(args.genomes_dir, "*.fasta"))

    records = []
    for fpath in fasta_files:
        genome_name = Path(fpath).stem
        sequence = ""
        with open(fpath) as f:
            for line in f:
                if not line.startswith(">"):
                    sequence += line.strip().upper()

        # Sliding window
        for start in range(0, len(sequence) - args.window, args.stride):
            fragment = sequence[start : start + args.window]
            if "N" * 10 in fragment:
                continue
            records.append({
                "sequence": fragment,
                "genome": genome_name,
                "gc_content": compute_gc_content(fragment),
                "start": start,
            })

    df = pd.DataFrame(records)
    os.makedirs(args.output, exist_ok=True)
    outpath = os.path.join(args.output, "pretrain_sequences_expanded.csv")
    df.to_csv(outpath, index=False)
    logger.info(f"Saved {len(df)} fragments to {outpath}")


if __name__ == "__main__":
    main()
