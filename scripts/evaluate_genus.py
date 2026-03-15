#!/usr/bin/env python3
"""
Genus-level evaluation for B-JEPA checkpoints.

Extracts CLS embeddings from the target encoder, then evaluates:
  1. Genus-level k-NN accuracy (k=1,5,10)
  2. Genus-level linear probe (logistic regression on frozen embeddings)
  3. Phylum-level k-NN and linear probe
  4. UMAP colored by genus, phylum, and GC content
  5. SVD spectrum and RankMe

Runs on CPU (except embedding extraction). Can run in a second terminal
while training continues.

Usage:
    python scripts/evaluate_genus.py \
        --checkpoint outputs/checkpoints/v7.0/epoch0002.pt \
        --data-path data/processed/pretrain_10M.csv \
        --taxonomy data/processed/genome_taxonomy.csv \
        --n-samples 30000 \
        --min-fragments 50 \
        --output-dir outputs/eval/v7.0_genus_epoch002
"""
import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


# ─── Dataset ────────────────────────────────────────────────────────────────
class GenomicEvalDataset(Dataset):
    """Load sequences with taxonomy labels for evaluation."""

    def __init__(self, csv_path, taxonomy_path, tokenizer_path,
                 max_len=512, min_fragments=50, max_per_genus=500,
                 n_samples=30000, seed=42):
        import csv as csv_mod
        import random

        random.seed(seed)
        np.random.seed(seed)

        # Load taxonomy: genome → {genus, phylum, species}
        print(f"  Loading taxonomy from {taxonomy_path}...")
        self.tax_map = {}
        with open(taxonomy_path) as f:
            reader = csv_mod.DictReader(f)
            for row in reader:
                self.tax_map[row["genome"]] = {
                    "species": row["species"],
                    "genus": row["genus"],
                    "phylum": row["phylum"],
                }

        # Load tokenizer
        print(f"  Loading tokenizer from {tokenizer_path}...")
        self.tokenizer = self._load_tokenizer(tokenizer_path)

        # Load data CSV — stream to avoid OOM on 10M+ rows
        print(f"  Scanning {csv_path} for genus-stratified sampling...")
        genus_to_rows = {}
        total_scanned = 0

        with open(csv_path) as f:
            reader = csv_mod.DictReader(f)
            for row in reader:
                genome = row["genome"]
                if genome not in self.tax_map:
                    continue

                genus = self.tax_map[genome]["genus"]
                if genus == "Unknown":
                    continue

                if genus not in genus_to_rows:
                    genus_to_rows[genus] = []
                genus_to_rows[genus].append(row)
                total_scanned += 1

                if total_scanned % 2_000_000 == 0:
                    print(f"    Scanned {total_scanned:,} rows...")

        # Filter to genera with enough fragments
        eligible_genera = {g: rows for g, rows in genus_to_rows.items()
                          if len(rows) >= min_fragments}
        print(f"  Genera with >={min_fragments} fragments: "
              f"{len(eligible_genera)}/{len(genus_to_rows)}")

        # Stratified sampling: up to max_per_genus per genus, total <= n_samples
        self.samples = []
        samples_per_genus = min(max_per_genus,
                                n_samples // max(len(eligible_genera), 1))

        for genus, rows in sorted(eligible_genera.items()):
            chosen = random.sample(rows, min(samples_per_genus, len(rows)))
            for row in chosen:
                self.samples.append({
                    "sequence": row["sequence"],
                    "gc": float(row["gc_content"]),
                    "genome": row["genome"],
                    "genus": genus,
                    "phylum": self.tax_map[row["genome"]]["phylum"],
                    "species": self.tax_map[row["genome"]]["species"],
                })

        # Trim to n_samples
        if len(self.samples) > n_samples:
            random.shuffle(self.samples)
            self.samples = self.samples[:n_samples]

        # Build label encodings
        self.genus_set = sorted(set(s["genus"] for s in self.samples))
        self.phylum_set = sorted(set(s["phylum"] for s in self.samples))
        self.genus_to_id = {g: i for i, g in enumerate(self.genus_set)}
        self.phylum_to_id = {p: i for i, p in enumerate(self.phylum_set)}

        self.max_len = max_len

        # Stats
        genus_counts = Counter(s["genus"] for s in self.samples)
        phylum_counts = Counter(s["phylum"] for s in self.samples)
        print(f"  Final: {len(self.samples)} samples, "
              f"{len(self.genus_set)} genera, {len(self.phylum_set)} phyla")
        print(f"  Top 10 genera: {genus_counts.most_common(10)}")
        print(f"  Phyla: {dict(phylum_counts.most_common())}")

    def _load_tokenizer(self, path):
        """Load BPE tokenizer from JSON."""
        try:
            from tokenizers import Tokenizer
            return Tokenizer.from_file(path)
        except ImportError:
            # Fallback: simple character-level tokenizer
            print("  WARNING: tokenizers library not found, using fallback")
            return None

    def _tokenize(self, seq):
        if self.tokenizer is not None:
            enc = self.tokenizer.encode(seq)
            ids = enc.ids[:self.max_len]
        else:
            # Fallback character tokenizer
            base_map = {"A": 5, "C": 6, "G": 7, "T": 8, "N": 9}
            ids = [base_map.get(c, 9) for c in seq[:self.max_len]]

        # Pad to max_len
        pad_len = self.max_len - len(ids)
        tokens = ids + [0] * pad_len  # 0 = PAD
        mask = [False] * len(ids) + [True] * pad_len
        return torch.tensor(tokens, dtype=torch.long), torch.tensor(mask, dtype=torch.bool)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        tokens, mask = self._tokenize(s["sequence"])
        return {
            "tokens": tokens,
            "mask": mask,
            "gc": s["gc"],
            "genus_id": self.genus_to_id[s["genus"]],
            "phylum_id": self.phylum_to_id[s["phylum"]],
            "genus": s["genus"],
            "phylum": s["phylum"],
        }


# ─── Model Loading ──────────────────────────────────────────────────────────
def load_model(checkpoint_path, device):
    """Load target encoder from B-JEPA checkpoint."""
    print(f"  Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Extract config
    cfg = ckpt.get("config", {})
    epoch = ckpt.get("epoch", "?")
    version = ckpt.get("version", "unknown")
    print(f"  Epoch: {epoch}, Version: {version}")

    embed_dim = cfg.get("embed_dim", 576)
    num_layers = cfg.get("num_layers", 12)
    num_heads = cfg.get("num_heads", 9)
    ff_dim = cfg.get("ff_dim", 2304)
    max_seq_len = cfg.get("max_seq_len", 512)
    vocab_size = cfg.get("vocab_size", 4096)

    print(f"  Architecture: {num_layers}L × {embed_dim}D × {num_heads}H")

    # Try importing from the project
    try:
        sys.path.insert(0, os.getcwd())
        from bdna_jepa.models.jepa_v6.pretrain_v6 import ContextEncoder
        encoder = ContextEncoder(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            ff_dim=ff_dim,
            max_seq_len=max_seq_len,
        )
    except Exception as e:
        print(f"  WARNING: Could not import ContextEncoder: {e}")
        print(f"  Attempting to build from checkpoint keys...")
        raise

    # Load target encoder weights (preferred for eval — EMA smoothed)
    # The checkpoint may store weights in different formats:
    #   A) Separate dicts: ckpt["target_encoder_state_dict"] = {"cls_token": ..., "layers.0...": ...}
    #   B) Single model dict: ckpt["model_state_dict"] = {"target_encoder.cls_token": ..., "context_encoder.cls_token": ...}
    #   C) Flat dict: ckpt keys directly include "target_encoder.cls_token", etc.

    def strip_prefix(state_dict, prefix):
        """Remove prefix from all keys in state_dict."""
        return {k[len(prefix):]: v for k, v in state_dict.items()
                if k.startswith(prefix)}

    loaded = False

    # Strategy A: dedicated key with clean state dict
    for key in ["target_encoder_state_dict", "target_encoder", "target_state_dict"]:
        if key in ckpt:
            sd = ckpt[key]
            # Check if keys need prefix stripping
            sample_key = next(iter(sd))
            if sample_key.startswith("target_encoder."):
                sd = strip_prefix(sd, "target_encoder.")
            elif sample_key.startswith("context_encoder."):
                sd = strip_prefix(sd, "context_encoder.")
            encoder.load_state_dict(sd)
            print(f"  Loaded TARGET encoder (EMA) from key '{key}'")
            loaded = True
            break

    # Strategy B: single model state dict with prefixed keys
    if not loaded:
        for key in ["model_state_dict", "state_dict", "context_encoder_state_dict",
                     "context_encoder", "encoder_state_dict"]:
            if key not in ckpt:
                continue
            sd = ckpt[key]
            sample_key = next(iter(sd))

            # Try target_encoder prefix first (EMA = better for eval)
            target_sd = strip_prefix(sd, "target_encoder.")
            if target_sd and len(target_sd) > 10:
                encoder.load_state_dict(target_sd)
                print(f"  Loaded TARGET encoder (EMA) from '{key}' (stripped prefix)")
                loaded = True
                break

            # Fall back to context_encoder prefix
            context_sd = strip_prefix(sd, "context_encoder.")
            if context_sd and len(context_sd) > 10:
                encoder.load_state_dict(context_sd)
                print(f"  Loaded CONTEXT encoder from '{key}' (stripped prefix)")
                loaded = True
                break

            # Try loading as-is (already clean keys)
            try:
                encoder.load_state_dict(sd)
                print(f"  Loaded encoder from '{key}' (direct)")
                loaded = True
                break
            except RuntimeError:
                continue

    # Strategy C: keys are at the top level of the checkpoint dict
    if not loaded:
        target_sd = strip_prefix(ckpt, "target_encoder.")
        if target_sd and len(target_sd) > 10:
            encoder.load_state_dict(target_sd)
            print(f"  Loaded TARGET encoder (EMA) from top-level keys")
            loaded = True

    if not loaded:
        context_sd = strip_prefix(ckpt, "context_encoder.")
        if context_sd and len(context_sd) > 10:
            encoder.load_state_dict(context_sd)
            print(f"  Loaded CONTEXT encoder from top-level keys")
            loaded = True

    if not loaded:
        raise KeyError(
            f"Could not load encoder. Checkpoint keys: {list(ckpt.keys())[:10]}... "
            f"Sample nested keys: {list(next(iter(v for v in ckpt.values() if isinstance(v, dict)), {}).keys())[:5]}"
        )

    encoder = encoder.to(device).eval()
    n_params = sum(p.numel() for p in encoder.parameters())
    print(f"  Parameters: {n_params:,}")

    return encoder, cfg


# ─── Embedding Extraction ──────────────────────────────────────────────────
@torch.no_grad()
def extract_embeddings(model, dataloader, device):
    """Extract CLS embeddings from encoder."""
    all_embeds = []
    all_gc = []
    all_genus = []
    all_phylum = []
    all_genus_names = []
    all_phylum_names = []

    for i, batch in enumerate(dataloader):
        tokens = batch["tokens"].to(device)
        mask = batch["mask"].to(device)

        # Forward — get CLS embeddings
        # The ContextEncoder.forward() signature varies across versions.
        # Try multiple calling conventions.
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            try:
                out = model(tokens, mask=mask)
            except TypeError:
                try:
                    out = model(tokens, pad_mask=mask)
                except TypeError:
                    try:
                        out = model(tokens, src_key_padding_mask=mask)
                    except TypeError:
                        # Last resort: just tokens, no mask
                        out = model(tokens)

        # Extract CLS: either direct tensor or dict
        if isinstance(out, dict):
            cls_emb = out.get("cls", out.get("context_cls", out.get("last_hidden_state", None)))
            if cls_emb is not None and cls_emb.dim() == 3:
                cls_emb = cls_emb[:, 0, :]  # first token = CLS
        elif isinstance(out, torch.Tensor):
            if out.dim() == 3:
                cls_emb = out[:, 0, :]
            else:
                cls_emb = out
        else:
            raise ValueError(f"Unexpected model output type: {type(out)}")

        all_embeds.append(cls_emb.float().cpu())
        all_gc.append(batch["gc"])
        all_genus.append(batch["genus_id"])
        all_phylum.append(batch["phylum_id"])
        all_genus_names.extend(batch["genus"])
        all_phylum_names.extend(batch["phylum"])

        if (i + 1) % 50 == 0:
            n = (i + 1) * dataloader.batch_size
            print(f"    Extracted {n} samples...")

    embeddings = torch.cat(all_embeds, dim=0).numpy()
    gc = torch.cat(all_gc, dim=0).numpy()
    genus_ids = torch.cat(all_genus, dim=0).numpy()
    phylum_ids = torch.cat(all_phylum, dim=0).numpy()

    print(f"  Total: {len(embeddings)} embeddings, dim={embeddings.shape[1]}")
    return embeddings, gc, genus_ids, phylum_ids, all_genus_names, all_phylum_names


# ─── Metrics ────────────────────────────────────────────────────────────────
def compute_rankme(embeddings):
    """RankMe: effective dimensionality via entropy of singular values."""
    U, S, Vt = np.linalg.svd(embeddings - embeddings.mean(0), full_matrices=False)
    S = S / S.sum()
    S = S[S > 1e-12]
    entropy = -np.sum(S * np.log(S))
    return float(np.exp(entropy))


def knn_accuracy(embeddings, labels, k_values=[1, 5, 10]):
    """k-NN accuracy using cosine similarity."""
    from sklearn.preprocessing import normalize

    # L2 normalize for cosine similarity
    X = normalize(embeddings, norm="l2")
    sim = X @ X.T
    np.fill_diagonal(sim, -1)  # exclude self

    results = {}
    for k in k_values:
        topk_idx = np.argsort(-sim, axis=1)[:, :k]
        topk_labels = labels[topk_idx]

        # Majority vote
        correct = 0
        for i in range(len(labels)):
            counts = Counter(topk_labels[i])
            pred = counts.most_common(1)[0][0]
            if pred == labels[i]:
                correct += 1

        results[k] = correct / len(labels)
    return results


def linear_probe(embeddings, labels, test_frac=0.2, seed=42):
    """Train logistic regression on frozen embeddings."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    n_classes = len(set(labels))
    if n_classes < 2:
        return {"accuracy": 0.0, "n_classes": n_classes}

    X_train, X_test, y_train, y_test = train_test_split(
        embeddings, labels, test_size=test_frac,
        random_state=seed, stratify=labels
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    clf = LogisticRegression(
        max_iter=1000,
        multi_class="multinomial",
        solver="lbfgs",
        C=1.0,
        random_state=seed,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    train_acc = clf.score(X_train, y_train)
    test_acc = clf.score(X_test, y_test)

    return {
        "train_accuracy": float(train_acc),
        "test_accuracy": float(test_acc),
        "n_classes": n_classes,
        "n_train": len(X_train),
        "n_test": len(X_test),
    }


# ─── Visualization ──────────────────────────────────────────────────────────
def generate_plots(embeddings, gc, genus_names, phylum_names,
                   genus_set, phylum_set, output_dir):
    """Generate UMAP and SVD plots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import matplotlib.cm as cm

    os.makedirs(output_dir, exist_ok=True)

    # ── SVD Spectrum ────────────────────────────────────────────────────
    print("  Generating SVD spectrum...")
    centered = embeddings - embeddings.mean(0)
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    cumvar = np.cumsum(S**2) / np.sum(S**2)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(S[:100], "o-", markersize=2, color="#2ecc71")
    axes[0].set_xlabel("Component")
    axes[0].set_ylabel("Singular Value")
    axes[0].set_title("SVD Spectrum (top 100)")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(cumvar, color="#e74c3c")
    axes[1].axhline(0.9, ls="--", color="gray", alpha=0.5, label="90%")
    axes[1].axhline(0.99, ls="--", color="gray", alpha=0.3, label="99%")
    axes[1].set_xlabel("Components")
    axes[1].set_ylabel("Cumulative Variance")
    axes[1].set_title("Cumulative Explained Variance")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "svd_spectrum.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {path}")

    # ── UMAP ────────────────────────────────────────────────────────────
    print("  Computing UMAP...")
    try:
        from umap import UMAP
    except ImportError:
        print("  WARNING: umap-learn not installed, skipping UMAP plots")
        return

    reducer = UMAP(n_components=2, n_neighbors=30, min_dist=0.3,
                   metric="cosine", random_state=42)
    umap_2d = reducer.fit_transform(embeddings)

    # ── UMAP by GC content ──────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 8))
    sc = ax.scatter(umap_2d[:, 0], umap_2d[:, 1], c=gc, cmap="RdYlBu_r",
                    s=3, alpha=0.4, rasterized=True)
    plt.colorbar(sc, ax=ax, label="GC Content", shrink=0.8)
    ax.set_title("UMAP — GC Content")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    path = os.path.join(output_dir, "umap_gc.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {path}")

    # ── UMAP by Phylum ──────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 8))
    phylum_list = np.array(phylum_names)
    unique_phyla = sorted(set(phylum_names))
    colors = cm.Set3(np.linspace(0, 1, max(len(unique_phyla), 12)))

    for i, phylum in enumerate(unique_phyla):
        mask = phylum_list == phylum
        ax.scatter(umap_2d[mask, 0], umap_2d[mask, 1],
                   c=[colors[i % len(colors)]], s=4, alpha=0.5,
                   label=f"{phylum} ({mask.sum()})", rasterized=True)

    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=7,
              markerscale=3, frameon=True)
    ax.set_title("UMAP — Phylum")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    path = os.path.join(output_dir, "umap_phylum.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {path}")

    # ── UMAP by Genus (top 30) ──────────────────────────────────────
    genus_list = np.array(genus_names)
    genus_counts = Counter(genus_names)
    top_genera = [g for g, c in genus_counts.most_common(30)]

    fig, ax = plt.subplots(figsize=(14, 10))

    # Plot "other" genera in gray first
    top_set = set(top_genera)
    other_mask = np.array([g not in top_set for g in genus_names])
    if other_mask.any():
        ax.scatter(umap_2d[other_mask, 0], umap_2d[other_mask, 1],
                   c="lightgray", s=2, alpha=0.15, label="Other",
                   rasterized=True)

    colors_genus = cm.tab20(np.linspace(0, 1, 20))
    colors_genus2 = cm.tab20b(np.linspace(0, 1, 10))
    all_colors = list(colors_genus) + list(colors_genus2)

    for i, genus in enumerate(top_genera):
        mask = genus_list == genus
        ax.scatter(umap_2d[mask, 0], umap_2d[mask, 1],
                   c=[all_colors[i % len(all_colors)]],
                   s=6, alpha=0.6,
                   label=f"{genus} ({mask.sum()})", rasterized=True)

    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=6,
              markerscale=3, frameon=True, ncol=1)
    ax.set_title("UMAP — Top 30 Genera")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    path = os.path.join(output_dir, "umap_genus_top30.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {path}")

    # ── t-SNE ───────────────────────────────────────────────────────
    print("  Computing t-SNE...")
    try:
        from sklearn.manifold import TSNE
        tsne = TSNE(n_components=2, perplexity=30, random_state=42,
                     n_iter=1000, init="pca")
        # Subsample for speed if needed
        n_tsne = min(len(embeddings), 10000)
        idx = np.random.RandomState(42).choice(len(embeddings), n_tsne, replace=False)
        tsne_2d = tsne.fit_transform(embeddings[idx])

        fig, ax = plt.subplots(figsize=(12, 8))
        sub_phyla = np.array(phylum_names)[idx]
        for i, phylum in enumerate(unique_phyla):
            mask = sub_phyla == phylum
            if mask.any():
                ax.scatter(tsne_2d[mask, 0], tsne_2d[mask, 1],
                           c=[colors[i % len(colors)]], s=4, alpha=0.5,
                           label=f"{phylum} ({mask.sum()})", rasterized=True)

        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=7,
                  markerscale=3, frameon=True)
        ax.set_title(f"t-SNE — Phylum (n={n_tsne})")
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        path = os.path.join(output_dir, "tsne_phylum.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"    Saved: {path}")
    except Exception as e:
        print(f"  t-SNE failed: {e}")


# ─── Main ───────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Genus-level evaluation for B-JEPA checkpoints")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to .pt checkpoint")
    parser.add_argument("--data-path", default="data/processed/pretrain_10M.csv",
                        help="Training data CSV")
    parser.add_argument("--taxonomy", default="data/processed/genome_taxonomy.csv",
                        help="Genome taxonomy CSV (from build_taxonomy.py)")
    parser.add_argument("--tokenizer-path",
                        default="data/tokenizer/bpe_4096.json",
                        help="BPE tokenizer path")
    parser.add_argument("--n-samples", type=int, default=30000,
                        help="Total samples to evaluate")
    parser.add_argument("--min-fragments", type=int, default=50,
                        help="Min fragments per genus to include")
    parser.add_argument("--max-per-genus", type=int, default=500,
                        help="Max fragments per genus")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--output-dir", required=True,
                        help="Output directory")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    t0 = time.time()

    # ── 1. Load model ───────────────────────────────────────────────────
    print("\n[1/6] Loading model...")
    encoder, cfg = load_model(args.checkpoint, device)

    # ── 2. Build dataset ────────────────────────────────────────────────
    print("\n[2/6] Building evaluation dataset...")
    dataset = GenomicEvalDataset(
        csv_path=args.data_path,
        taxonomy_path=args.taxonomy,
        tokenizer_path=args.tokenizer_path,
        max_len=cfg.get("max_seq_len", 512),
        min_fragments=args.min_fragments,
        max_per_genus=args.max_per_genus,
        n_samples=args.n_samples,
    )

    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=True,
    )

    # ── 3. Extract embeddings ───────────────────────────────────────────
    print("\n[3/6] Extracting embeddings...")
    embeddings, gc, genus_ids, phylum_ids, genus_names, phylum_names = \
        extract_embeddings(encoder, loader, device)

    # Free GPU memory
    del encoder
    torch.cuda.empty_cache()

    # ── 4. Compute metrics ──────────────────────────────────────────────
    print("\n[4/6] Computing metrics...")
    t_metrics = time.time()

    rankme = compute_rankme(embeddings)
    norms = np.linalg.norm(embeddings, axis=1)
    per_dim_std = embeddings.std(axis=0).mean()
    print(f"  RankMe:      {rankme:.1f}")
    print(f"  Norm:        {norms.mean():.1f} ± {norms.std():.2f}")
    print(f"  Per-dim std: {per_dim_std:.3f}")

    # Genus-level k-NN
    print(f"\n  Genus-level k-NN ({len(dataset.genus_set)} classes)...")
    genus_knn = knn_accuracy(embeddings, genus_ids, k_values=[1, 5, 10, 20])
    for k, acc in genus_knn.items():
        print(f"    k={k:<3d}  accuracy={acc:.4f} ({acc*100:.1f}%)")

    # Phylum-level k-NN
    print(f"\n  Phylum-level k-NN ({len(dataset.phylum_set)} classes)...")
    phylum_knn = knn_accuracy(embeddings, phylum_ids, k_values=[1, 5, 10])
    for k, acc in phylum_knn.items():
        print(f"    k={k:<3d}  accuracy={acc:.4f} ({acc*100:.1f}%)")

    # Genus-level linear probe
    print(f"\n  Genus-level linear probe...")
    genus_probe = linear_probe(embeddings, genus_ids)
    print(f"    Train accuracy: {genus_probe['train_accuracy']:.4f} "
          f"({genus_probe['train_accuracy']*100:.1f}%)")
    print(f"    Test accuracy:  {genus_probe['test_accuracy']:.4f} "
          f"({genus_probe['test_accuracy']*100:.1f}%)")

    # Phylum-level linear probe
    print(f"\n  Phylum-level linear probe...")
    phylum_probe = linear_probe(embeddings, phylum_ids)
    print(f"    Train accuracy: {phylum_probe['train_accuracy']:.4f} "
          f"({phylum_probe['train_accuracy']*100:.1f}%)")
    print(f"    Test accuracy:  {phylum_probe['test_accuracy']:.4f} "
          f"({phylum_probe['test_accuracy']*100:.1f}%)")

    # GC content correlation with embeddings (sanity check)
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    X_tr, X_te, y_tr, y_te = train_test_split(
        embeddings, gc, test_size=0.2, random_state=42)
    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr)
    X_te_s = sc.transform(X_te)
    gc_ridge = Ridge(alpha=1.0).fit(X_tr_s, y_tr)
    gc_r2 = gc_ridge.score(X_te_s, y_te)
    print(f"\n  GC content R² (Ridge): {gc_r2:.4f}")

    print(f"\n  Metrics computed in {time.time() - t_metrics:.1f}s")

    # ── 5. Generate plots ───────────────────────────────────────────────
    print("\n[5/6] Generating plots...")
    generate_plots(embeddings, gc, genus_names, phylum_names,
                   dataset.genus_set, dataset.phylum_set, args.output_dir)

    # ── 6. Save summary ────────────────────────────────────────────────
    print("\n[6/6] Saving summary...")
    summary = {
        "checkpoint": str(args.checkpoint),
        "n_samples": len(embeddings),
        "embed_dim": embeddings.shape[1],
        "n_genera": len(dataset.genus_set),
        "n_phyla": len(dataset.phylum_set),
        "rankme": rankme,
        "norm_mean": float(norms.mean()),
        "norm_std": float(norms.std()),
        "per_dim_std": float(per_dim_std),
        "gc_r2": gc_r2,
        "genus_knn": {str(k): float(v) for k, v in genus_knn.items()},
        "phylum_knn": {str(k): float(v) for k, v in phylum_knn.items()},
        "genus_probe": genus_probe,
        "phylum_probe": phylum_probe,
        "genera_list": dataset.genus_set,
        "phyla_list": dataset.phylum_set,
        "wall_time_seconds": time.time() - t0,
    }

    os.makedirs(args.output_dir, exist_ok=True)
    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: {summary_path}")

    # ── Final Report ────────────────────────────────────────────────────
    print(f"""
{'='*60}
  B-JEPA Genus-Level Evaluation
{'='*60}
  Checkpoint:    {args.checkpoint}
  Samples:       {len(embeddings)}
  Genera:        {len(dataset.genus_set)}
  Phyla:         {len(dataset.phylum_set)}

  RankMe:        {rankme:.1f}
  GC R²:         {gc_r2:.4f}

  Genus k-NN:    k=1  {genus_knn[1]*100:.1f}%
                 k=5  {genus_knn[5]*100:.1f}%
                 k=10 {genus_knn[10]*100:.1f}%

  Genus Probe:   train={genus_probe['train_accuracy']*100:.1f}%  test={genus_probe['test_accuracy']*100:.1f}%

  Phylum k-NN:   k=1  {phylum_knn[1]*100:.1f}%
                 k=5  {phylum_knn[5]*100:.1f}%

  Phylum Probe:  train={phylum_probe['train_accuracy']*100:.1f}%  test={phylum_probe['test_accuracy']*100:.1f}%

  Wall time:     {time.time() - t0:.0f}s
  Output:        {args.output_dir}
{'='*60}""")


if __name__ == "__main__":
    main()
