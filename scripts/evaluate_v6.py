#!/usr/bin/env python3
"""
B-JEPA v6.x — Quick evaluation: species UMAP, k-NN, linear probe.
Run in a separate terminal while training continues.

Usage:
    python scripts/evaluate_v6.py \
        --checkpoint outputs/checkpoints/v6.2/epoch0005.pt \
        --data-path data/processed/pretrain_2M.csv \
        --tokenizer-path data/tokenizer/bpe_4096.json \
        --output-dir outputs/eval/v6.2_epoch005 \
        --n-samples 5000
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ══════════════════════════════════════════════════════════════════════════
# 1. LOAD MODEL FROM CHECKPOINT
# ══════════════════════════════════════════════════════════════════════════

def load_v6_model(checkpoint_path: str, device: str):
    """Load BJEPAv6 model from checkpoint (config is stored inside)."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from bdna_jepa.models.jepa_v6.pretrain_v6 import BJEPAv6

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]

    print(f"Checkpoint: epoch {ckpt.get('epoch', '?')}, version {ckpt.get('version', '?')}")
    print(f"  embed_dim={cfg['embed_dim']}, layers={cfg['num_layers']}, heads={cfg['num_heads']}")

    # Build model from saved config
    model = BJEPAv6(
        vocab_size=cfg.get("vocab_size", 4096),
        embed_dim=cfg["embed_dim"],
        num_layers=cfg["num_layers"],
        num_heads=cfg["num_heads"],
        ff_dim=cfg["ff_dim"],
        max_seq_len=cfg["max_seq_len"],
        predictor_dim=cfg.get("predictor_dim", 384),
        predictor_depth=cfg.get("predictor_depth", 6),
        predictor_heads=cfg.get("predictor_heads", 6),
        var_gamma=cfg.get("var_gamma", 1.0),
    )

    # Load weights
    state_dict = ckpt["model_state_dict"]
    cleaned = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(cleaned, strict=False)
    model.to(device).eval()

    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    return model, cfg


# ══════════════════════════════════════════════════════════════════════════
# 2. EXTRACT EMBEDDINGS
# ══════════════════════════════════════════════════════════════════════════

def load_species_map(species_map_path):
    """Load genome -> species mapping from CSV."""
    import csv
    mapping = {}
    with open(species_map_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            mapping[row["genome"]] = row["species"]
    print(f"  Loaded {len(mapping)} genome->species mappings")
    return mapping


def build_dataloader(data_path, tokenizer_path, max_seq_len, n_samples,
                     batch_size=64, species_map=None):
    """Build dataloader from pretrain CSV."""
    from bdna_jepa.data.tokenizer import BPETokenizer
    from bdna_jepa.data.dataset import BacterialGenomeDataset

    tokenizer = BPETokenizer(tokenizer_path)
    dataset = BacterialGenomeDataset(
        csv_path=data_path,
        tokenizer=tokenizer,
        max_length=max_seq_len,
    )

    n = min(n_samples, len(dataset))
    subset = Subset(dataset, list(range(n)))

    def collate(batch):
        max_len = max(item["tokens"].shape[0] for item in batch)
        padded = torch.zeros(len(batch), max_len, dtype=batch[0]["tokens"].dtype)
        for i, item in enumerate(batch):
            L = item["tokens"].shape[0]
            padded[i, :L] = item["tokens"]

        # Resolve species from genome ID if species_map provided
        species_list = []
        for item in batch:
            sp = item.get("species", "")
            if (not sp or sp == "") and species_map:
                genome = item.get("genome", "")
                sp = species_map.get(genome, "")
            species_list.append(sp)

        return {
            "input_ids": padded,
            "gc_content": torch.tensor([item.get("gc_content", 0.5) for item in batch]),
            "species": species_list,
            "genome": [item.get("genome", "") for item in batch],
        }

    loader = DataLoader(subset, batch_size=batch_size, shuffle=False,
                        num_workers=2, pin_memory=True, collate_fn=collate)
    return loader


@torch.no_grad()
def extract_embeddings(model, dataloader, device):
    """Extract CLS embeddings from target encoder (EMA)."""
    all_cls = []
    all_gc = []
    all_species = []

    encoder = model.target_encoder

    for i, batch in enumerate(dataloader):
        tokens = batch["input_ids"].to(device)
        B, L = tokens.shape

        # ContextEncoder expects (tokens, position_ids)
        # For eval (no masking), position_ids = arange(L) for each sample
        position_ids = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)

        out = encoder(tokens, position_ids)

        if isinstance(out, dict):
            cls = out.get("cls", out.get("cls_token"))
            if cls is None:
                tok = out.get("tokens", out.get("x"))
                cls = tok[:, 0, :]
        elif isinstance(out, torch.Tensor):
            if out.dim() == 3:
                cls = out[:, 0, :]
            else:
                cls = out
        else:
            raise ValueError(f"Unexpected output type: {type(out)}")

        all_cls.append(cls.cpu())
        all_gc.append(batch["gc_content"])
        all_species.extend(batch["species"])

        if (i + 1) % 20 == 0:
            print(f"  Extracted {(i+1) * dataloader.batch_size} samples...")

    embeddings = torch.cat(all_cls, dim=0)
    gc = torch.cat(all_gc, dim=0)
    print(f"  Total: {embeddings.shape[0]} embeddings, dim={embeddings.shape[1]}")
    return embeddings.numpy(), gc.numpy(), np.array(all_species)


# ══════════════════════════════════════════════════════════════════════════
# 3. EVALUATION METRICS
# ══════════════════════════════════════════════════════════════════════════

def compute_rankme(embeddings):
    centered = embeddings - embeddings.mean(axis=0)
    _, S, _ = np.linalg.svd(centered, full_matrices=False)
    p = S / S.sum()
    p = p[p > 1e-10]
    entropy = -(p * np.log(p)).sum()
    return float(np.exp(entropy))


def run_knn(embeddings, labels, k_values=(1, 5, 10, 20)):
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()
    y = le.fit_transform(labels)
    n_classes = len(le.classes_)
    print(f"  k-NN: {len(y)} samples, {n_classes} classes")

    results = {}
    for k in k_values:
        if k >= len(y):
            continue
        knn = KNeighborsClassifier(n_neighbors=k, metric="cosine")
        cv = min(5, n_classes)
        if cv < 2:
            print(f"  Skipping k-NN (only {n_classes} class)")
            return {}
        scores = cross_val_score(knn, embeddings, y, cv=cv, scoring="accuracy")
        results[k] = {"mean": float(scores.mean()), "std": float(scores.std())}
        print(f"    k={k}: {scores.mean():.4f} +/- {scores.std():.4f}")
    return results


def run_linear_probe(embeddings, labels):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.pipeline import Pipeline

    le = LabelEncoder()
    y = le.fit_transform(labels)
    n_classes = len(le.classes_)
    print(f"  Linear probe: {len(y)} samples, {n_classes} classes")

    if n_classes < 2:
        print("  Skipping (only 1 class)")
        return {}

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, C=1.0, n_jobs=-1)),
    ])
    cv = min(5, n_classes)
    scores = cross_val_score(pipe, embeddings, y, cv=cv, scoring="accuracy")
    result = {"accuracy": float(scores.mean()), "std": float(scores.std())}
    print(f"    Accuracy: {scores.mean():.4f} +/- {scores.std():.4f}")
    return result


# ══════════════════════════════════════════════════════════════════════════
# 4. VISUALIZATION
# ══════════════════════════════════════════════════════════════════════════

def plot_species_umap(embeddings, species, gc, output_dir, n_max=5000):
    from umap import UMAP

    n = min(n_max, len(embeddings))
    emb = embeddings[:n]
    sp = species[:n]
    gc_sub = gc[:n]

    reducer = UMAP(n_neighbors=30, min_dist=0.3, random_state=42, n_jobs=1)
    coords = reducer.fit_transform(emb)

    # --- UMAP colored by species (top N most common) ---
    unique, counts = np.unique(sp, return_counts=True)
    top_k = min(20, len(unique))
    top_species = unique[np.argsort(-counts)[:top_k]]
    top_set = set(top_species)

    fig, ax = plt.subplots(figsize=(12, 9))
    # Plot "other" first (gray)
    other_mask = np.array([s not in top_set for s in sp])
    if other_mask.any():
        ax.scatter(coords[other_mask, 0], coords[other_mask, 1],
                   s=3, c="lightgray", alpha=0.3, label="other", rasterized=True)

    cmap = plt.cm.tab20
    for i, s in enumerate(top_species):
        mask = sp == s
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   s=8, c=[cmap(i / top_k)], alpha=0.6, label=s[:30], rasterized=True)

    ax.set_title(f"UMAP — Top {top_k} Species (n={n})")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=6, markerscale=3)
    fig.tight_layout()
    path = output_dir / "umap_species.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")

    # --- UMAP colored by GC content ---
    fig, ax = plt.subplots(figsize=(10, 8))
    sc = ax.scatter(coords[:, 0], coords[:, 1], s=3, c=gc_sub,
                    cmap="coolwarm", alpha=0.5, rasterized=True)
    plt.colorbar(sc, ax=ax, label="GC content")
    ax.set_title(f"UMAP — GC Content (n={n})")
    fig.tight_layout()
    path = output_dir / "umap_gc.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")

    return coords


def plot_svd(embeddings, output_dir):
    centered = embeddings - embeddings.mean(axis=0)
    _, S, _ = np.linalg.svd(centered, full_matrices=False)
    cumvar = np.cumsum(S ** 2) / np.sum(S ** 2)
    rankme = compute_rankme(embeddings)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.semilogy(range(1, len(S) + 1), S, "b-", linewidth=1)
    ax1.set_xlabel("Singular value index")
    ax1.set_ylabel("Singular value (log)")
    ax1.set_title("SVD Spectrum")
    ax1.grid(True, alpha=0.3)

    ax2.plot(range(1, len(cumvar) + 1), cumvar, "r-", linewidth=1)
    ax2.axhline(y=0.9, color="gray", linestyle="--", alpha=0.5, label="90%")
    ax2.axhline(y=0.99, color="gray", linestyle=":", alpha=0.5, label="99%")
    ax2.set_xlabel("Number of components")
    ax2.set_ylabel("Cumulative variance explained")
    ax2.set_title(f"Cumulative Variance (RankMe={rankme:.1f})")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    path = output_dir / "svd_spectrum.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════
# 5. MAIN
# ══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="B-JEPA v6.x Evaluation")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-path", type=str, default="data/processed/pretrain_2M.csv")
    parser.add_argument("--tokenizer-path", type=str, default="data/tokenizer/bpe_4096.json")
    parser.add_argument("--species-map", type=str, default="data/processed/genome_species.csv",
                        help="CSV with genome,species columns")
    parser.add_argument("--output-dir", type=str, default="outputs/eval/v6.2")
    parser.add_argument("--n-samples", type=int, default=5000)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--skip-knn", action="store_true", help="Skip k-NN (slow)")
    parser.add_argument("--skip-probe", action="store_true", help="Skip linear probe")
    args = parser.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load model ──
    print("\n[1/5] Loading model...")
    model, cfg = load_v6_model(args.checkpoint, device)

    # ── Load species mapping ──
    species_map = None
    if args.species_map and Path(args.species_map).exists():
        print("\n  Loading species map...")
        species_map = load_species_map(args.species_map)
    else:
        print(f"\n  No species map at {args.species_map} — species eval will be skipped")

    # ── Build dataloader ──
    print("\n[2/5] Building dataloader...")
    loader = build_dataloader(
        args.data_path, args.tokenizer_path,
        cfg["max_seq_len"], args.n_samples,
        species_map=species_map,
    )

    # ── Extract embeddings ──
    print("\n[3/5] Extracting embeddings...")
    t0 = time.time()
    embeddings, gc, species = extract_embeddings(model, loader, device)
    print(f"  Done in {time.time() - t0:.1f}s")

    # ── Metrics ──
    print("\n[4/5] Computing metrics...")
    rankme = compute_rankme(embeddings)
    print(f"  RankMe: {rankme:.1f}")

    norms = np.linalg.norm(embeddings, axis=1)
    print(f"  Norm: {norms.mean():.1f} +/- {norms.std():.2f}")

    stds = embeddings.std(axis=0)
    print(f"  Per-dim std: {stds.mean():.3f}")

    # Filter to samples with species labels
    has_species = np.array([s != "" and s != "unknown" for s in species])
    n_labeled = has_species.sum()
    print(f"  Samples with species labels: {n_labeled}/{len(species)}")

    knn_results = {}
    probe_results = {}

    if n_labeled > 50:
        emb_labeled = embeddings[has_species]
        sp_labeled = species[has_species]

        if not args.skip_knn:
            print("\n  Running k-NN species classification...")
            knn_results = run_knn(emb_labeled, sp_labeled)

        if not args.skip_probe:
            print("\n  Running linear probe...")
            probe_results = run_linear_probe(emb_labeled, sp_labeled)
    else:
        print("  Not enough labeled samples for k-NN / linear probe")

    # ── Plots ──
    print("\n[5/5] Generating plots...")
    plot_species_umap(embeddings, species, gc, output_dir)
    plot_svd(embeddings, output_dir)

    # ── Save summary ──
    summary = {
        "checkpoint": args.checkpoint,
        "n_samples": len(embeddings),
        "embed_dim": embeddings.shape[1],
        "rankme": rankme,
        "mean_norm": float(norms.mean()),
        "mean_per_dim_std": float(stds.mean()),
        "n_species": len(set(species[has_species])) if n_labeled > 0 else 0,
        "knn": knn_results,
        "linear_probe": probe_results,
    }
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Summary saved: {summary_path}")

    print(f"\n{'='*60}")
    print(f"  RankMe:        {rankme:.1f}")
    if knn_results:
        best_k = max(knn_results, key=lambda k: knn_results[k]["mean"])
        print(f"  Best k-NN:     k={best_k}, acc={knn_results[best_k]['mean']:.4f}")
    if probe_results:
        print(f"  Linear probe:  {probe_results['accuracy']:.4f}")
    print(f"  All saved to:  {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
