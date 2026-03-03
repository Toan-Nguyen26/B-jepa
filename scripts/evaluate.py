#!/usr/bin/env python3
"""B-JEPA evaluation pipeline.

Usage:
    python scripts/evaluate.py --checkpoint outputs/checkpoints/v4.0/best.pt --version v4.0
    python scripts/evaluate.py --checkpoint outputs/checkpoints/v4.0/best.pt --data data.csv --output results/
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from torch.utils.data import DataLoader

from bdna_jepa.config import V31_CONFIG, V40_CONFIG
from bdna_jepa.hub import load_encoder
from bdna_jepa.data.tokenizer import get_tokenizer
from bdna_jepa.data.dataset import BacterialGenomeDataset, collate_fn
from bdna_jepa.utils.metrics import compute_rankme, compute_feature_std, compute_spectral_analysis
from bdna_jepa.utils.logging import get_logger


logger = get_logger(__name__)


def extract_embeddings(
    encoder, dataset, device, batch_size=256, max_samples=5000,
) -> tuple[np.ndarray, dict]:
    """Extract CLS embeddings from encoder."""
    from torch.utils.data import Subset

    n = min(len(dataset), max_samples)
    subset = Subset(dataset, range(n))
    loader = DataLoader(subset, batch_size=batch_size, collate_fn=collate_fn)

    all_emb, all_meta = [], {"gc_content": [], "species": []}
    encoder.eval()

    with torch.no_grad():
        for batch in loader:
            tokens = batch["tokens"].to(device)
            mask = batch.get("attention_mask")
            if mask is not None:
                mask = mask.to(device)
            cls = encoder.encode(tokens, mask)
            all_emb.append(cls.cpu().numpy())
            if "gc_content" in batch:
                all_meta["gc_content"].extend(batch["gc_content"].tolist())

    embeddings = np.concatenate(all_emb, axis=0)
    return embeddings, all_meta


def main():
    parser = argparse.ArgumentParser(description="B-JEPA Evaluation")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--version", type=str, default="v4.0", choices=["v3.1", "v4.0"])
    parser.add_argument("--data", type=str, default="data/processed/pretrain_sequences_expanded.csv")
    parser.add_argument("--output", type=str, default="outputs/eval/")
    parser.add_argument("--max-samples", type=int, default=5000)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output, exist_ok=True)

    # Load encoder
    encoder = load_encoder(args.checkpoint, args.version, device=str(device))
    logger.info(f"Loaded encoder from {args.checkpoint}")

    # Dataset
    tokenizer = get_tokenizer(args.version)
    config = V40_CONFIG if args.version == "v4.0" else V31_CONFIG
    dataset = BacterialGenomeDataset(args.data, tokenizer, max_length=config.encoder.max_seq_len)

    # Extract embeddings
    embeddings, meta = extract_embeddings(encoder, dataset, device, max_samples=args.max_samples)
    logger.info(f"Extracted {embeddings.shape[0]} embeddings of dim {embeddings.shape[1]}")

    # Spectral analysis
    spectral = compute_spectral_analysis(torch.from_numpy(embeddings))
    results = {
        "rankme": spectral["effective_rank"],
        "feature_std": spectral["feature_std"],
        "power_law_alpha": spectral["power_law_alpha"],
        "top1_explained": spectral["top1_explained"],
        "n_samples": embeddings.shape[0],
        "embed_dim": embeddings.shape[1],
    }

    # GC regression
    if meta["gc_content"]:
        from bdna_jepa.evaluation.eval import gc_regression
        gc_vals = np.array(meta["gc_content"][:embeddings.shape[0]])
        gc_result = gc_regression(embeddings[:len(gc_vals)], gc_vals)
        results["gc_r2"] = gc_result["r2"]
        logger.info(f"GC R2: {gc_result['r2']:.4f}")

    # Save
    results_path = os.path.join(args.output, "eval_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {results_path}")

    # Spectral plot
    try:
        from bdna_jepa.evaluation.eval import plot_spectral_analysis
        plot_path = os.path.join(args.output, "spectral.pdf")
        plot_spectral_analysis(embeddings, plot_path)
        logger.info(f"Spectral plot saved to {plot_path}")
    except ImportError:
        logger.warning("matplotlib not available, skipping plots")

    logger.info(f"RankMe={results['rankme']:.1f} | std={results['feature_std']:.4f}")


if __name__ == "__main__":
    main()
