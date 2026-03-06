#!/usr/bin/env python3
"""
B-JEPA v4.0 — Post-training evaluation & figure generation
============================================================
Run on the Vast.ai instance after stopping training.

Usage:
    python evaluate_v4.py \
        --checkpoint outputs/checkpoints/v4.0/epoch0039.pt \
        --data data/processed/pretrain_500K.csv \
        --tokenizer data/tokenizer/bpe_4096.json \
        --config configs/training/v4.0.yaml \
        --output_dir outputs/figures/v4.0 \
        --log_file logs/training.log       # optional: for training curves

Generates:
    1. fig1_training_curves.png        — MLM/JEPA/total loss + RankMe over epochs
    2. fig2_umap_embeddings.png        — UMAP of CLS embeddings (colored by GC%)
    3. fig3_singular_values.png        — Singular value spectrum + RankMe annotation
    4. fig4_feature_distributions.png  — Per-dimension std + activation histograms
    5. fig5_mlm_accuracy.png           — Token prediction accuracy breakdown
    6. fig6_embedding_health.png       — Cosine similarity distribution + norm histogram
    7. summary_metrics.json            — All numeric metrics in one place
"""

import argparse
import json
import re
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── Attempt optional imports ──────────────────────────────────────────────
try:
    from umap import UMAP
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print("WARNING: umap-learn not installed. UMAP figure will be skipped.")
    print("  pip install umap-learn")

try:
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("WARNING: scikit-learn not installed. Linear probe will be skipped.")
    print("  pip install scikit-learn")


# ══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════

FIGURE_DPI = 200
FIGSIZE_WIDE = (14, 5)
FIGSIZE_SQUARE = (8, 7)
FIGSIZE_TALL = (10, 12)
N_EVAL_SAMPLES = 6000       # samples for embedding analysis
N_UMAP_SAMPLES = 5000       # samples for UMAP (subset for speed)
BATCH_SIZE = 64
SEED = 42
PALETTE = {
    "blue": "#2563eb",
    "orange": "#ea580c",
    "green": "#16a34a",
    "purple": "#9333ea",
    "red": "#dc2626",
    "gray": "#6b7280",
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
})


# ══════════════════════════════════════════════════════════════════════════
# 0. LOADING UTILITIES
# ══════════════════════════════════════════════════════════════════════════

def load_model_and_tokenizer(checkpoint_path: str, config_path: str, tokenizer_path: str, device: str):
    """Load BJEPA model from checkpoint using BJEPAConfig."""
    from bdna_jepa.config import BJEPAConfig
    from bdna_jepa.models.jepa import BJEPA
    from bdna_jepa.data.tokenizer import CharTokenizer

    # Load config from YAML
    import yaml
    import dataclasses
    import bdna_jepa.config as config_module
    with open(config_path) as f:
        raw = yaml.safe_load(f)

    # Collect all dataclass types from config module
    config_classes = {}
    for name in dir(config_module):
        obj = getattr(config_module, name)
        if dataclasses.is_dataclass(obj) and isinstance(obj, type):
            config_classes[name] = obj

    def build_dataclass(cls, data):
        """Recursively build a dataclass from a dict."""
        if not isinstance(data, dict):
            return data
        type_hints = {f.name: f.type for f in dataclasses.fields(cls)}
        kwargs = {}
        for key, value in data.items():
            if key in type_hints:
                type_name = type_hints[key] if isinstance(type_hints[key], str) else getattr(type_hints[key], '__name__', str(type_hints[key]))
                if isinstance(value, dict) and type_name in config_classes:
                    kwargs[key] = build_dataclass(config_classes[type_name], value)
                else:
                    kwargs[key] = value
        return cls(**kwargs)

    config = build_dataclass(BJEPAConfig, raw)
    print(f"Config loaded: encoder.embed_dim={config.encoder.embed_dim}, max_seq_len={config.encoder.max_seq_len}")

    # Build full BJEPA model (has context_encoder + mlm_head)
    model = BJEPA(config)

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)

    # Strip "module." prefix if present (DDP)
    cleaned = {}
    for k, v in state_dict.items():
        cleaned[k.replace("module.", "")] = v
    model.load_state_dict(cleaned, strict=False)
    model.to(device).eval()

    epoch = ckpt.get("epoch", "?")
    loss = ckpt.get("loss", 0.0)
    print(f"Loaded checkpoint: epoch {epoch}, loss {loss:.4f}")
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    # Load tokenizer — try BPE first, fall back to CharTokenizer
    try:
        from bdna_jepa.data.tokenizer import BPETokenizer
        tokenizer = BPETokenizer(tokenizer_path)
        print(f"Tokenizer: BPE from {tokenizer_path}, vocab={tokenizer.vocab_size}")
    except (ImportError, AttributeError):
        tokenizer = CharTokenizer()
        print("Tokenizer: CharTokenizer (fallback)")

    return model, tokenizer, config


def pad_collate(batch, pad_id=0):
    """Collate with padding for variable-length sequences."""
    max_len = max(item["tokens"].shape[0] for item in batch)
    padded_batch = {}
    # Pad tokens
    padded = torch.full((len(batch), max_len), pad_id, dtype=batch[0]["tokens"].dtype)
    for i, item in enumerate(batch):
        L = item["tokens"].shape[0]
        padded[i, :L] = item["tokens"]
    padded_batch["input_ids"] = padded  # rename to input_ids for consistency

    # Collect gc_content
    padded_batch["gc_content"] = torch.tensor([item.get("gc_content", 0.5) for item in batch])
    # Collect species
    padded_batch["species"] = [item.get("species", "") for item in batch]
    return padded_batch


def build_dataloader(data_path: str, tokenizer, max_seq_len: int, n_samples: int):
    """Build a dataloader for evaluation."""
    from bdna_jepa.data.dataset import BacterialGenomeDataset

    dataset = BacterialGenomeDataset(
        csv_path=data_path,
        tokenizer=tokenizer,
        max_length=max_seq_len,
    )

    # Subsample
    indices = list(range(min(n_samples, len(dataset))))
    subset = torch.utils.data.Subset(dataset, indices)

    loader = DataLoader(
        subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        drop_last=False,
        collate_fn=pad_collate,
    )
    return loader


# ══════════════════════════════════════════════════════════════════════════
# 1. EXTRACT EMBEDDINGS
# ══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def extract_embeddings(model, dataloader, device, max_batches=None):
    """Extract CLS embeddings from the BJEPA model's context encoder."""
    cls_embeddings = []
    all_gc_contents = []

    encoder = model.context_encoder

    for i, batch in enumerate(dataloader):
        if max_batches and i >= max_batches:
            break

        input_ids = batch["input_ids"].to(device)
        out = encoder(input_ids)

        # Handle different return types: dict or tensor
        if isinstance(out, dict):
            cls_emb = out.get("cls", out.get("cls_token"))
            if cls_emb is None:
                tokens = out.get("tokens", out.get("x"))
                cls_emb = tokens[:, 0, :]
        elif isinstance(out, torch.Tensor):
            cls_emb = out[:, 0, :]
        else:
            raise ValueError(f"Unexpected encoder output type: {type(out)}")

        cls_embeddings.append(cls_emb.cpu())

        # Use real GC content from dataset
        gc = batch.get("gc_content", torch.full((input_ids.shape[0],), 0.5))
        if isinstance(gc, torch.Tensor):
            all_gc_contents.append(gc.cpu())
        else:
            all_gc_contents.append(torch.tensor(gc))

        if (i + 1) % 20 == 0:
            print(f"  Extracted {(i+1) * BATCH_SIZE} samples...")

    cls_embeddings = torch.cat(cls_embeddings, dim=0)  # (N, D)
    all_gc_contents = torch.cat(all_gc_contents, dim=0)

    print(f"  Total: {cls_embeddings.shape[0]} embeddings, dim={cls_embeddings.shape[1]}")
    return cls_embeddings, all_gc_contents


def compute_gc_from_ids(input_ids: torch.Tensor) -> torch.Tensor:
    """
    Approximate GC content from tokenized sequences.
    Since BPE tokens encode character-level DNA, we just look at the raw ids.
    This is a rough heuristic — adjust token-to-nucleotide mapping if needed.
    Returns tensor of shape (B,) with GC fraction per sequence.
    """
    # Fallback: use sequence length variance as a proxy feature
    # A proper implementation would decode tokens back to DNA
    B = input_ids.shape[0]
    gc = torch.zeros(B)
    for i in range(B):
        ids = input_ids[i]
        # Use token ID distribution as a proxy for sequence content
        # Hash-based GC proxy: sum of token IDs mod some prime
        non_special = ids[ids > 4]  # skip special tokens
        if len(non_special) > 0:
            gc[i] = (non_special.float().mean() % 100) / 100.0
        else:
            gc[i] = 0.5
    return gc


# ══════════════════════════════════════════════════════════════════════════
# 2. MLM ACCURACY
# ══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_mlm(model, dataloader, device, mask_ratio=0.15, n_batches=50):
    """
    Evaluate MLM accuracy by masking tokens and checking predictions.
    """
    from bdna_jepa.data.masking import random_mask

    correct = 0
    total = 0
    loss_sum = 0.0

    encoder = model.context_encoder
    mlm_head = model.mlm_head

    for i, batch in enumerate(dataloader):
        if i >= n_batches:
            break

        input_ids = batch["input_ids"].to(device)
        B, L = input_ids.shape

        # Apply masking (returns masked_tokens, mask, labels)
        masked_ids, mask, labels = random_mask(
            input_ids, mask_ratio=mask_ratio, mask_id=1,
            vocab_size=4096, special_token_max=5,
        )

        # Forward through encoder
        out = encoder(masked_ids)

        # Get token embeddings
        if isinstance(out, dict):
            token_embs = out.get("tokens", out.get("x"))
            if token_embs is None:
                # If encoder returns (B, L+1, D) with CLS, skip CLS
                token_embs = list(out.values())[0]
        elif isinstance(out, torch.Tensor):
            # (B, L+1, D) — skip CLS token at position 0
            token_embs = out[:, 1:, :]  # (B, L, D)
        else:
            raise ValueError(f"Unexpected output type: {type(out)}")

        # Ensure token_embs matches input length
        if token_embs.shape[1] != L:
            # CLS prepended — skip first token
            token_embs = token_embs[:, 1:L+1, :]

        # MLM head prediction
        logits = mlm_head(token_embs)  # (B, L, vocab_size)

        # Only evaluate at masked positions (labels != -100)
        valid = labels != -100
        if valid.sum() == 0:
            continue

        masked_logits = logits[valid]  # (N_masked, vocab_size)
        masked_labels = labels[valid]  # (N_masked,)

        preds = masked_logits.argmax(dim=-1)
        correct += (preds == masked_labels).sum().item()
        total += masked_labels.numel()

        loss = F.cross_entropy(masked_logits, masked_labels)
        loss_sum += loss.item()

    if total == 0:
        print("  WARNING: No masked tokens found")
        return None

    accuracy = correct / total
    avg_loss = loss_sum / min(n_batches, i + 1)

    print(f"  MLM Accuracy: {accuracy:.4f} ({correct}/{total})")
    print(f"  MLM Loss: {avg_loss:.4f} (vs ln(810)={np.log(810):.4f} random baseline)")

    return {
        "accuracy": accuracy,
        "loss": avg_loss,
        "correct": correct,
        "total": total,
        "random_baseline_loss": float(np.log(810)),
    }


# ══════════════════════════════════════════════════════════════════════════
# 3. EMBEDDING ANALYSIS FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════

def compute_rankme(embeddings: torch.Tensor) -> float:
    """Compute RankMe (effective rank via entropy of singular values)."""
    # Center
    centered = embeddings - embeddings.mean(dim=0)
    # SVD
    _, S, _ = torch.svd(centered)
    # Normalize to probability distribution
    p = S / S.sum()
    p = p[p > 1e-10]
    # Shannon entropy
    entropy = -(p * torch.log(p)).sum()
    rankme = torch.exp(entropy).item()
    return rankme, S.numpy()


def compute_embedding_health(embeddings: torch.Tensor):
    """Compute embedding health metrics."""
    norms = torch.norm(embeddings, dim=1)
    stds = embeddings.std(dim=0)
    feature_std = stds.mean().item()

    # Cosine similarity matrix (subsample for speed)
    n = min(1000, embeddings.shape[0])
    sub = F.normalize(embeddings[:n], dim=1)
    cosine_sim = (sub @ sub.T).numpy()
    # Remove diagonal
    mask = ~np.eye(n, dtype=bool)
    off_diag = cosine_sim[mask]

    return {
        "norms": norms.numpy(),
        "feature_stds": stds.numpy(),
        "mean_feature_std": feature_std,
        "cosine_similarities": off_diag,
        "mean_cosine_sim": float(off_diag.mean()),
        "mean_norm": float(norms.mean()),
        "std_norm": float(norms.std()),
    }


def compute_dead_neurons(model, dataloader, device, n_batches=20):
    """Check for dead neurons in SwiGLU layers."""
    activation_counts = {}
    encoder = model.context_encoder

    def hook_fn(name):
        def fn(module, input, output):
            if name not in activation_counts:
                activation_counts[name] = {
                    "total": 0,
                    "active": None,
                }
            inp = input[0] if isinstance(input, tuple) else input
            active = (inp.abs() > 1e-6).float().sum(dim=(0, 1))
            if activation_counts[name]["active"] is None:
                activation_counts[name]["active"] = active.cpu()
            else:
                activation_counts[name]["active"] += active.cpu()
            activation_counts[name]["total"] += inp.shape[0] * inp.shape[1]
        return fn

    hooks = []
    for name, module in encoder.named_modules():
        if "w2" in name and isinstance(module, torch.nn.Linear):
            hooks.append(module.register_forward_hook(hook_fn(name)))

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= n_batches:
                break
            input_ids = batch["input_ids"].to(device)
            encoder(input_ids)

    for h in hooks:
        h.remove()

    dead_fracs = {}
    for name, counts in activation_counts.items():
        frac_active = counts["active"] / counts["total"]
        dead = (frac_active < 0.01).float().mean().item()
        dead_fracs[name] = dead

    overall_dead = np.mean(list(dead_fracs.values())) if dead_fracs else 0.0
    print(f"  Dead neuron fraction: {overall_dead:.4f}")
    return dead_fracs, overall_dead


# ══════════════════════════════════════════════════════════════════════════
# 4. PARSE TRAINING LOG
# ══════════════════════════════════════════════════════════════════════════

def parse_training_log(log_path: str):
    """Parse training log to extract loss curves and eval metrics."""
    epochs = []
    steps = []

    step_data = defaultdict(list)  # step -> {mlm, jepa, loss}
    epoch_data = []  # [{epoch, avg_loss}]
    eval_data = []   # [{epoch, rankme, std, loss}]

    step_pattern = re.compile(
        r"Step (\d+) \| epoch (\d+) \| loss=([\d.]+) \| mlm=([\d.]+) \| jepa=([\d.]+) \| lr=([\d.e+-]+)"
    )
    epoch_pattern = re.compile(r"Epoch (\d+) complete \| avg_loss=([\d.]+)")
    eval_pattern = re.compile(r"Eval \| RankMe=([\d.]+) \| std=([\d.]+) \| loss=([\d.]+)")

    seen_steps = set()
    seen_epochs = set()
    seen_evals = set()

    with open(log_path) as f:
        for line in f:
            # Step-level
            m = step_pattern.search(line)
            if m:
                step = int(m.group(1))
                if step not in seen_steps:
                    seen_steps.add(step)
                    step_data["step"].append(step)
                    step_data["epoch"].append(int(m.group(2)))
                    step_data["loss"].append(float(m.group(3)))
                    step_data["mlm"].append(float(m.group(4)))
                    step_data["jepa"].append(float(m.group(5)))
                    step_data["lr"].append(float(m.group(6)))

            # Epoch-level
            m = epoch_pattern.search(line)
            if m:
                ep = int(m.group(1))
                if ep not in seen_epochs:
                    seen_epochs.add(ep)
                    epoch_data.append({
                        "epoch": ep,
                        "avg_loss": float(m.group(2)),
                    })

            # Eval
            m = eval_pattern.search(line)
            if m:
                key = m.group(0)
                if key not in seen_evals:
                    seen_evals.add(key)
                    eval_data.append({
                        "rankme": float(m.group(1)),
                        "std": float(m.group(2)),
                        "eval_loss": float(m.group(3)),
                    })

    # Assign eval epochs (eval happens every 5 epochs starting at epoch 1)
    eval_epochs = [1, 4, 9, 14, 19, 24, 29, 34, 39, 44]
    for i, ed in enumerate(eval_data):
        if i < len(eval_epochs):
            ed["epoch"] = eval_epochs[i]

    print(f"  Parsed: {len(seen_steps)} steps, {len(epoch_data)} epochs, {len(eval_data)} evals")
    return step_data, epoch_data, eval_data


# ══════════════════════════════════════════════════════════════════════════
# 5. FIGURE GENERATION
# ══════════════════════════════════════════════════════════════════════════

def fig1_training_curves(step_data, epoch_data, eval_data, output_dir):
    """Training curves: MLM loss, JEPA loss, RankMe, feature_std."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    steps = np.array(step_data["step"])
    mlm = np.array(step_data["mlm"])
    jepa = np.array(step_data["jepa"])
    total = np.array(step_data["loss"])
    lr = np.array(step_data["lr"])

    # Smooth for readability
    def smooth(y, window=20):
        if len(y) < window:
            return y
        kernel = np.ones(window) / window
        return np.convolve(y, kernel, mode="valid")

    steps_s = steps[:len(smooth(mlm))]

    # (0,0) MLM Loss
    ax = axes[0, 0]
    ax.plot(steps_s, smooth(mlm), color=PALETTE["blue"], linewidth=1.2)
    ax.axhline(y=np.log(810), color=PALETTE["red"], linestyle="--", alpha=0.7, label=f"Random baseline (ln810={np.log(810):.2f})")
    ax.axhline(y=np.log(4096), color=PALETTE["gray"], linestyle=":", alpha=0.5, label=f"Full vocab random (ln4096={np.log(4096):.2f})")
    ax.set_title("MLM Loss")
    ax.set_xlabel("Step")
    ax.set_ylabel("Cross-entropy loss")
    ax.legend(fontsize=8)

    # (0,1) JEPA Loss
    ax = axes[0, 1]
    ax.plot(steps_s, smooth(jepa), color=PALETTE["orange"], linewidth=1.2)
    ax.set_title("JEPA Loss (MSE)")
    ax.set_xlabel("Step")
    ax.set_ylabel("MSE")

    # (0,2) Total Loss
    ax = axes[0, 2]
    ax.plot(steps_s, smooth(total), color=PALETTE["purple"], linewidth=1.2)
    ax.set_title("Total Weighted Loss")
    ax.set_xlabel("Step")

    # (1,0) RankMe over evals
    ax = axes[1, 0]
    if eval_data:
        eval_epochs = [e["epoch"] for e in eval_data]
        rankmes = [e["rankme"] for e in eval_data]
        ax.plot(eval_epochs, rankmes, "o-", color=PALETTE["green"], linewidth=2, markersize=6)
        ax.axhline(y=576, color=PALETTE["gray"], linestyle=":", alpha=0.5, label="Max (d_model=576)")
        ax.set_title("RankMe (Effective Rank)")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("RankMe")
        ax.legend(fontsize=8)

    # (1,1) Feature std over evals
    ax = axes[1, 1]
    if eval_data:
        feature_stds = [e["std"] for e in eval_data]
        ax.plot(eval_epochs, feature_stds, "s-", color=PALETTE["red"], linewidth=2, markersize=6)
        ax.set_title("Feature Std (mean across dims)")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Std")

    # (1,2) Learning rate schedule
    ax = axes[1, 2]
    ax.plot(steps, lr, color=PALETTE["gray"], linewidth=1)
    ax.set_title("Learning Rate Schedule")
    ax.set_xlabel("Step")
    ax.set_ylabel("LR")

    fig.suptitle("B-JEPA v4.0 Training Curves — 500K Bacterial Genomes", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = output_dir / "fig1_training_curves.png"
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def fig2_umap_embeddings(embeddings, gc_contents, output_dir):
    """UMAP of CLS embeddings colored by GC content proxy."""
    if not HAS_UMAP:
        print("  Skipping UMAP (umap-learn not installed)")
        return

    n = min(N_UMAP_SAMPLES, embeddings.shape[0])
    emb = embeddings[:n].numpy()
    gc = gc_contents[:n].numpy()

    print(f"  Running UMAP on {n} samples...")
    reducer = UMAP(n_neighbors=30, min_dist=0.3, metric="cosine", random_state=SEED)
    coords = reducer.fit_transform(emb)

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_WIDE)

    # Left: colored by GC content proxy
    ax = axes[0]
    sc = ax.scatter(coords[:, 0], coords[:, 1], c=gc, cmap="viridis",
                    s=3, alpha=0.5, rasterized=True)
    plt.colorbar(sc, ax=ax, label="GC Content (proxy)")
    ax.set_title("UMAP — Colored by GC Content")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")

    # Right: colored by embedding norm
    norms = np.linalg.norm(emb, axis=1)
    ax = axes[1]
    sc = ax.scatter(coords[:, 0], coords[:, 1], c=norms, cmap="magma",
                    s=3, alpha=0.5, rasterized=True)
    plt.colorbar(sc, ax=ax, label="Embedding L2 Norm")
    ax.set_title("UMAP — Colored by Embedding Norm")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")

    fig.suptitle("B-JEPA v4.0 CLS Embedding Space", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = output_dir / "fig2_umap_embeddings.png"
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def fig3_singular_values(embeddings, output_dir):
    """Singular value spectrum with RankMe annotation."""
    rankme, S = compute_rankme(embeddings)

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_WIDE)

    # Left: Raw singular values
    ax = axes[0]
    ax.bar(range(len(S)), S, color=PALETTE["blue"], alpha=0.7, width=1.0)
    ax.axvline(x=rankme, color=PALETTE["red"], linestyle="--", linewidth=2,
               label=f"RankMe = {rankme:.1f}")
    ax.set_title("Singular Value Spectrum")
    ax.set_xlabel("Component index")
    ax.set_ylabel("Singular value")
    ax.legend()

    # Right: Cumulative variance explained
    ax = axes[1]
    variance = S ** 2
    cum_var = np.cumsum(variance) / variance.sum()
    ax.plot(range(len(cum_var)), cum_var, color=PALETTE["green"], linewidth=2)
    ax.axhline(y=0.95, color=PALETTE["red"], linestyle="--", alpha=0.7, label="95% variance")
    ax.axhline(y=0.99, color=PALETTE["orange"], linestyle=":", alpha=0.7, label="99% variance")
    n95 = np.searchsorted(cum_var, 0.95) + 1
    n99 = np.searchsorted(cum_var, 0.99) + 1
    ax.axvline(x=n95, color=PALETTE["red"], linestyle="--", alpha=0.3)
    ax.axvline(x=n99, color=PALETTE["orange"], linestyle=":", alpha=0.3)
    ax.set_title(f"Cumulative Variance (95% at {n95}, 99% at {n99} dims)")
    ax.set_xlabel("Number of components")
    ax.set_ylabel("Cumulative variance explained")
    ax.legend()

    fig.suptitle(f"B-JEPA v4.0 — Embedding Dimensionality (RankMe={rankme:.1f}/{embeddings.shape[1]})",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = output_dir / "fig3_singular_values.png"
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return rankme


def fig4_feature_distributions(embeddings, health, output_dir):
    """Per-dimension statistics and activation distributions."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # (0,0) Per-dimension std (sorted)
    ax = axes[0, 0]
    stds_sorted = np.sort(health["feature_stds"])[::-1]
    ax.bar(range(len(stds_sorted)), stds_sorted, color=PALETTE["blue"], alpha=0.7, width=1.0)
    ax.axhline(y=health["mean_feature_std"], color=PALETTE["red"], linestyle="--",
               label=f"Mean={health['mean_feature_std']:.4f}")
    ax.set_title("Per-Dimension Std (sorted)")
    ax.set_xlabel("Dimension (sorted by std)")
    ax.set_ylabel("Standard deviation")
    ax.legend()

    # (0,1) Histogram of a few dimensions
    ax = axes[0, 1]
    emb_np = embeddings.numpy()
    for dim_idx in [0, 100, 200, 400]:
        if dim_idx < emb_np.shape[1]:
            ax.hist(emb_np[:, dim_idx], bins=50, alpha=0.4, label=f"dim {dim_idx}")
    ax.set_title("Activation Distribution (selected dims)")
    ax.set_xlabel("Activation value")
    ax.set_ylabel("Count")
    ax.legend(fontsize=8)

    # (1,0) Embedding norms histogram
    ax = axes[1, 0]
    ax.hist(health["norms"], bins=80, color=PALETTE["purple"], alpha=0.7)
    ax.axvline(x=health["mean_norm"], color=PALETTE["red"], linestyle="--",
               label=f"Mean={health['mean_norm']:.2f}")
    ax.set_title("Embedding L2 Norm Distribution")
    ax.set_xlabel("L2 Norm")
    ax.set_ylabel("Count")
    ax.legend()

    # (1,1) Cosine similarity distribution
    ax = axes[1, 1]
    ax.hist(health["cosine_similarities"], bins=100, color=PALETTE["orange"], alpha=0.7)
    ax.axvline(x=health["mean_cosine_sim"], color=PALETTE["red"], linestyle="--",
               label=f"Mean={health['mean_cosine_sim']:.4f}")
    ax.set_title("Pairwise Cosine Similarity")
    ax.set_xlabel("Cosine similarity")
    ax.set_ylabel("Count")
    ax.legend()

    fig.suptitle("B-JEPA v4.0 — Embedding Health", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = output_dir / "fig4_feature_distributions.png"
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def fig5_mlm_accuracy(mlm_results, output_dir):
    """MLM accuracy summary figure."""
    if mlm_results is None:
        print("  Skipping MLM figure (no results)")
        return

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_WIDE)

    # Left: Loss comparison
    ax = axes[0]
    labels = ["Model", "Random\n(active vocab)", "Random\n(full vocab)"]
    values = [mlm_results["loss"], np.log(810), np.log(4096)]
    colors = [PALETTE["blue"], PALETTE["red"], PALETTE["gray"]]
    bars = ax.bar(labels, values, color=colors, alpha=0.8, width=0.5)
    ax.set_title("MLM Loss Comparison")
    ax.set_ylabel("Cross-entropy loss")
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f"{val:.3f}", ha="center", fontsize=10)

    # Right: Accuracy gauge
    ax = axes[1]
    acc = mlm_results["accuracy"]
    random_acc = 1.0 / 810
    ax.barh(["Model", "Random\n(1/810)"], [acc, random_acc], color=[PALETTE["green"], PALETTE["gray"]], height=0.4)
    ax.set_title("MLM Token Prediction Accuracy")
    ax.set_xlabel("Accuracy")
    ax.set_xlim(0, max(acc * 1.3, 0.01))
    for i, v in enumerate([acc, random_acc]):
        ax.text(v + 0.0005, i, f"{v:.4f}", va="center", fontsize=10)

    fig.suptitle("B-JEPA v4.0 — Masked Language Modeling Performance", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = output_dir / "fig5_mlm_accuracy.png"
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def fig6_embedding_health_summary(health, rankme, dead_frac, mlm_results, output_dir):
    """Dashboard-style summary of all health metrics."""
    fig = plt.figure(figsize=(12, 6))
    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.4, wspace=0.4)

    metrics = [
        ("RankMe", f"{rankme:.1f} / 576", rankme / 576),
        ("Feature Std", f"{health['mean_feature_std']:.4f}", min(health['mean_feature_std'] / 0.5, 1.0)),
        ("Mean Cosine Sim", f"{health['mean_cosine_sim']:.4f}", 1.0 - abs(health['mean_cosine_sim'])),
        ("Dead Neurons", f"{dead_frac:.2%}", 1.0 - dead_frac),
        ("Mean Norm", f"{health['mean_norm']:.2f}", min(health['mean_norm'] / 20, 1.0)),
        ("Norm Std", f"{health['std_norm']:.3f}", min(health['std_norm'] / 5, 1.0)),
        ("MLM Acc", f"{mlm_results['accuracy']:.4f}" if mlm_results else "N/A",
         mlm_results['accuracy'] * 10 if mlm_results else 0),
        ("MLM Loss", f"{mlm_results['loss']:.3f}" if mlm_results else "N/A",
         1.0 - mlm_results['loss'] / np.log(4096) if mlm_results else 0),
    ]

    for i, (name, value, score) in enumerate(metrics):
        row, col = divmod(i, 4)
        ax = fig.add_subplot(gs[row, col])

        # Simple gauge
        color = PALETTE["green"] if score > 0.5 else PALETTE["orange"] if score > 0.2 else PALETTE["red"]
        ax.barh([0], [score], color=color, alpha=0.7, height=0.6)
        ax.barh([0], [1.0], color=PALETTE["gray"], alpha=0.1, height=0.6)
        ax.set_xlim(0, 1.0)
        ax.set_yticks([])
        ax.set_xticks([])
        ax.set_title(name, fontsize=11, fontweight="bold")
        ax.text(0.5, -0.5, value, transform=ax.transAxes, ha="center", fontsize=13)

    fig.suptitle("B-JEPA v4.0 — Evaluation Dashboard (Epoch 39)",
                 fontsize=14, fontweight="bold")
    path = output_dir / "fig6_embedding_health.png"
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════
# 6. MAIN
# ══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="B-JEPA v4.0 Evaluation")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to checkpoint .pt file")
    parser.add_argument("--data", type=str, required=True,
                        help="Path to pretrain CSV")
    parser.add_argument("--tokenizer", type=str, required=True,
                        help="Path to BPE tokenizer JSON")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to training config YAML")
    parser.add_argument("--output_dir", type=str, default="outputs/figures/v4.0",
                        help="Output directory for figures")
    parser.add_argument("--log_file", type=str, default=None,
                        help="Training log file for curves (optional)")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: auto, cuda, cpu")
    parser.add_argument("--n_samples", type=int, default=N_EVAL_SAMPLES,
                        help="Number of samples for embedding extraction")
    args = parser.parse_args()

    # Device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"Device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load model ──
    print("\n[1/7] Loading model...")
    model, tokenizer, config = load_model_and_tokenizer(
        args.checkpoint, args.config, args.tokenizer, device
    )

    # BJEPAConfig is a dataclass — access encoder config
    max_seq_len = getattr(config.encoder, "max_seq_len", 512) if hasattr(config, "encoder") else 512

    # ── Build dataloader ──
    print("\n[2/7] Building dataloader...")
    loader = build_dataloader(args.data, tokenizer, max_seq_len, args.n_samples)

    # ── Extract embeddings ──
    print("\n[3/7] Extracting embeddings...")
    embeddings, gc_contents = extract_embeddings(model, loader, device)

    # ── Compute metrics ──
    print("\n[4/7] Computing embedding health...")
    health = compute_embedding_health(embeddings)
    rankme, singular_values = compute_rankme(embeddings)
    print(f"  RankMe: {rankme:.1f}")
    print(f"  Mean feature std: {health['mean_feature_std']:.4f}")
    print(f"  Mean cosine sim: {health['mean_cosine_sim']:.4f}")
    print(f"  Mean norm: {health['mean_norm']:.2f} +/- {health['std_norm']:.2f}")

    print("\n[4b/7] Computing dead neurons...")
    loader2 = build_dataloader(args.data, tokenizer, max_seq_len, 2000)
    dead_fracs, dead_frac = compute_dead_neurons(model, loader2, device)

    # ── MLM accuracy ──
    print("\n[5/7] Evaluating MLM accuracy...")
    loader3 = build_dataloader(args.data, tokenizer, max_seq_len, 5000)
    mlm_results = evaluate_mlm(model, loader3, device)

    # ── Generate figures ──
    print("\n[6/7] Generating figures...")

    # Fig 1: Training curves (needs log file)
    if args.log_file and Path(args.log_file).exists():
        print("  Parsing training log...")
        step_data, epoch_data, eval_data = parse_training_log(args.log_file)
        fig1_training_curves(step_data, epoch_data, eval_data, output_dir)
    else:
        print("  Skipping training curves (no log file provided)")

    # Fig 2: UMAP
    fig2_umap_embeddings(embeddings, gc_contents, output_dir)

    # Fig 3: Singular values
    fig3_singular_values(embeddings, output_dir)

    # Fig 4: Feature distributions
    fig4_feature_distributions(embeddings, health, output_dir)

    # Fig 5: MLM accuracy
    fig5_mlm_accuracy(mlm_results, output_dir)

    # Fig 6: Dashboard
    fig6_embedding_health_summary(health, rankme, dead_frac, mlm_results, output_dir)

    # ── Save summary JSON ──
    print("\n[7/7] Saving summary metrics...")
    summary = {
        "checkpoint": args.checkpoint,
        "n_samples": embeddings.shape[0],
        "d_model": embeddings.shape[1],
        "rankme": rankme,
        "mean_feature_std": health["mean_feature_std"],
        "mean_cosine_similarity": health["mean_cosine_sim"],
        "mean_norm": health["mean_norm"],
        "std_norm": health["std_norm"],
        "dead_neuron_frac": dead_frac,
        "mlm": mlm_results,
    }
    summary_path = output_dir / "summary_metrics.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  Saved: {summary_path}")

    print(f"\n{'='*60}")
    print(f"  All figures saved to: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
