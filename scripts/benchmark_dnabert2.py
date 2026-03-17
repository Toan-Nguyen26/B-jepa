#!/usr/bin/env python3
"""
Benchmark DNABERT-2 on the same genus evaluation dataset used for B-JEPA.
Direct comparison: same data, same metrics, different encoder.

Usage:
    pip install transformers --break-system-packages
    python scripts/benchmark_dnabert2.py \
        --data-path data/processed/eval_20k.csv \
        --taxonomy data/processed/genome_taxonomy.csv \
        --n-samples 5000 \
        --min-fragments 200 \
        --max-per-genus 200 \
        --device cpu \
        --output-dir outputs/eval/dnabert2_baseline
"""
import argparse
import csv
import json
import os
import random
import time
from collections import Counter

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


# ─── Dataset (reuses same sampling logic as B-JEPA eval) ───────────────────
class GenomicEvalDataset(Dataset):
    def __init__(self, csv_path, taxonomy_path, max_len=512,
                 min_fragments=50, max_per_genus=500, n_samples=30000, seed=42):
        random.seed(seed)
        np.random.seed(seed)

        # Load taxonomy
        self.tax_map = {}
        with open(taxonomy_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.tax_map[row["genome"]] = {
                    "species": row["species"],
                    "genus": row["genus"],
                    "phylum": row["phylum"],
                }

        # Load and filter data
        genus_to_rows = {}
        with open(csv_path) as f:
            reader = csv.DictReader(f)
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

        eligible = {g: rows for g, rows in genus_to_rows.items()
                    if len(rows) >= min_fragments}
        print(f"  Genera with >={min_fragments} fragments: {len(eligible)}")

        # Stratified sampling
        self.samples = []
        per_genus = min(max_per_genus, n_samples // max(len(eligible), 1))
        for genus, rows in sorted(eligible.items()):
            chosen = random.sample(rows, min(per_genus, len(rows)))
            for row in chosen:
                self.samples.append({
                    "sequence": row["sequence"],
                    "gc": float(row["gc_content"]),
                    "genus": genus,
                    "phylum": self.tax_map[row["genome"]]["phylum"],
                })

        if len(self.samples) > n_samples:
            random.shuffle(self.samples)
            self.samples = self.samples[:n_samples]

        self.genus_set = sorted(set(s["genus"] for s in self.samples))
        self.phylum_set = sorted(set(s["phylum"] for s in self.samples))
        self.genus_to_id = {g: i for i, g in enumerate(self.genus_set)}
        self.phylum_to_id = {p: i for i, p in enumerate(self.phylum_set)}
        self.max_len = max_len

        print(f"  Final: {len(self.samples)} samples, "
              f"{len(self.genus_set)} genera, {len(self.phylum_set)} phyla")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            "sequence": s["sequence"][:self.max_len * 6],  # raw DNA, tokenize later
            "gc": s["gc"],
            "genus_id": self.genus_to_id[s["genus"]],
            "phylum_id": self.phylum_to_id[s["phylum"]],
            "genus": s["genus"],
            "phylum": s["phylum"],
        }


# ─── Metrics (same as B-JEPA eval) ─────────────────────────────────────────
def compute_rankme(embeddings):
    U, S, Vt = np.linalg.svd(embeddings - embeddings.mean(0), full_matrices=False)
    S = S / S.sum()
    S = S[S > 1e-12]
    return float(np.exp(-np.sum(S * np.log(S))))


def knn_accuracy(embeddings, labels, k_values=[1, 5, 10, 20]):
    from sklearn.preprocessing import normalize
    X = normalize(embeddings, norm="l2")
    sim = X @ X.T
    np.fill_diagonal(sim, -1)
    results = {}
    for k in k_values:
        topk_idx = np.argsort(-sim, axis=1)[:, :k]
        topk_labels = labels[topk_idx]
        correct = 0
        for i in range(len(labels)):
            counts = Counter(topk_labels[i])
            pred = counts.most_common(1)[0][0]
            if pred == labels[i]:
                correct += 1
        results[k] = correct / len(labels)
    return results


def linear_probe(embeddings, labels, test_frac=0.2, seed=42, min_per_class=5):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    label_counts = Counter(labels)
    valid_classes = {c for c, n in label_counts.items() if n >= min_per_class}
    if len(valid_classes) < 2:
        return {"test_accuracy": 0.0, "n_classes": len(valid_classes)}

    mask = np.array([l in valid_classes for l in labels])
    X, y = embeddings[mask], labels[mask]
    unique = sorted(set(y))
    label_map = {old: new for new, old in enumerate(unique)}
    y = np.array([label_map[l] for l in y])

    n_test = int(len(y) * test_frac)
    use_stratify = n_test >= len(unique)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_frac, random_state=seed,
        stratify=y if use_stratify else None)

    sc = StandardScaler()
    X_tr, X_te = sc.fit_transform(X_tr), sc.transform(X_te)

    clf = LogisticRegression(max_iter=1000, solver="lbfgs", C=1.0,
                             random_state=seed)
    clf.fit(X_tr, y_tr)
    return {
        "train_accuracy": float(clf.score(X_tr, y_tr)),
        "test_accuracy": float(clf.score(X_te, y_te)),
        "n_classes": len(unique),
        "n_train": len(X_tr),
        "n_test": len(X_te),
    }


# ─── Main ───────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Benchmark DNABERT-2 baseline")
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--taxonomy", required=True)
    parser.add_argument("--n-samples", type=int, default=5000)
    parser.add_argument("--min-fragments", type=int, default=200)
    parser.add_argument("--max-per-genus", type=int, default=200)
    parser.add_argument("--max-len", type=int, default=512,
                        help="Max tokens (DNABERT-2 BPE)")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")
    t0 = time.time()

    # ── 1. Load DNABERT-2 ──────────────────────────────────────────────
    print("\n[1/5] Loading DNABERT-2...")
    from transformers import AutoTokenizer, AutoModel, AutoConfig

    model_name = "zhihan1996/DNABERT-2-117M"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Patch config — newer transformers versions require pad_token_id
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    if not hasattr(config, "pad_token_id") or config.pad_token_id is None:
        config.pad_token_id = tokenizer.pad_token_id or 0

    model = AutoModel.from_pretrained(model_name, config=config, trust_remote_code=True)
    model = model.to(device).eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {model_name}")
    print(f"  Parameters: {n_params:,}")
    print(f"  Embed dim: {model.config.hidden_size}")

    # ── 2. Build dataset ───────────────────────────────────────────────
    print("\n[2/5] Building evaluation dataset...")
    dataset = GenomicEvalDataset(
        csv_path=args.data_path,
        taxonomy_path=args.taxonomy,
        max_len=args.max_len,
        min_fragments=args.min_fragments,
        max_per_genus=args.max_per_genus,
        n_samples=args.n_samples,
    )

    # ── 3. Extract embeddings ──────────────────────────────────────────
    print("\n[3/5] Extracting DNABERT-2 embeddings...")
    all_embeds = []
    all_gc = []
    all_genus = []
    all_phylum = []

    # Process in batches
    batch_seqs = []
    batch_meta = []

    for i, sample in enumerate(dataset.samples):
        batch_seqs.append(sample["sequence"][:2048])  # limit sequence length
        batch_meta.append(sample)

        if len(batch_seqs) == args.batch_size or i == len(dataset.samples) - 1:
            # Tokenize batch
            encoded = tokenizer(
                batch_seqs,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_len,
            ).to(device)

            with torch.no_grad():
                with torch.autocast("cuda", dtype=torch.bfloat16,
                                    enabled=device.type == "cuda"):
                    outputs = model(**encoded)

            # Mean pool over non-padded tokens (better than CLS for DNABERT-2)
            hidden = outputs.last_hidden_state  # [B, L, D]
            mask = encoded["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1)

            all_embeds.append(pooled.float().cpu().numpy())
            for m in batch_meta:
                all_gc.append(m["gc"])
                all_genus.append(dataset.genus_to_id[m["genus"]])
                all_phylum.append(dataset.phylum_to_id[m["phylum"]])

            batch_seqs = []
            batch_meta = []

            if (i + 1) % 200 == 0:
                print(f"    Extracted {i + 1}/{len(dataset.samples)} samples...")

    embeddings = np.concatenate(all_embeds, axis=0)
    gc = np.array(all_gc)
    genus_ids = np.array(all_genus)
    phylum_ids = np.array(all_phylum)
    print(f"  Total: {len(embeddings)} embeddings, dim={embeddings.shape[1]}")

    # ── 4. Compute metrics ─────────────────────────────────────────────
    print("\n[4/5] Computing metrics...")
    rankme = compute_rankme(embeddings)
    norms = np.linalg.norm(embeddings, axis=1)
    print(f"  RankMe:      {rankme:.1f}")
    print(f"  Norm:        {norms.mean():.1f} ± {norms.std():.2f}")

    print(f"\n  Genus k-NN ({len(dataset.genus_set)} classes)...")
    genus_knn = knn_accuracy(embeddings, genus_ids)
    for k, acc in genus_knn.items():
        print(f"    k={k:<3d}  accuracy={acc:.4f} ({acc*100:.1f}%)")

    print(f"\n  Phylum k-NN ({len(dataset.phylum_set)} classes)...")
    phylum_knn = knn_accuracy(embeddings, phylum_ids)
    for k, acc in phylum_knn.items():
        print(f"    k={k:<3d}  accuracy={acc:.4f} ({acc*100:.1f}%)")

    print(f"\n  Genus linear probe...")
    genus_probe = linear_probe(embeddings, genus_ids)
    print(f"    Train: {genus_probe['train_accuracy']*100:.1f}%  "
          f"Test: {genus_probe['test_accuracy']*100:.1f}%")

    print(f"\n  Phylum linear probe...")
    phylum_probe = linear_probe(embeddings, phylum_ids)
    print(f"    Train: {phylum_probe['train_accuracy']*100:.1f}%  "
          f"Test: {phylum_probe['test_accuracy']*100:.1f}%")

    # GC R²
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    X_tr, X_te, y_tr, y_te = train_test_split(embeddings, gc, test_size=0.2, random_state=42)
    sc = StandardScaler()
    gc_r2 = Ridge(alpha=1.0).fit(sc.fit_transform(X_tr), y_tr).score(sc.transform(X_te), y_te)
    print(f"\n  GC R²: {gc_r2:.4f}")

    # ── 5. Save ────────────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    summary = {
        "model": model_name,
        "n_params": n_params,
        "embed_dim": int(embeddings.shape[1]),
        "n_samples": len(embeddings),
        "n_genera": len(dataset.genus_set),
        "n_phyla": len(dataset.phylum_set),
        "rankme": rankme,
        "gc_r2": gc_r2,
        "genus_knn": {str(k): float(v) for k, v in genus_knn.items()},
        "phylum_knn": {str(k): float(v) for k, v in phylum_knn.items()},
        "genus_probe": genus_probe,
        "phylum_probe": phylum_probe,
        "genera_list": dataset.genus_set,
        "wall_time": time.time() - t0,
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"""
{'='*60}
  DNABERT-2 Baseline
{'='*60}
  Model:         {model_name}
  Parameters:    {n_params:,}
  Samples:       {len(embeddings)}
  Genera:        {len(dataset.genus_set)}

  RankMe:        {rankme:.1f}
  GC R²:         {gc_r2:.4f}

  Genus k-NN:    k=1  {genus_knn[1]*100:.1f}%
                 k=5  {genus_knn[5]*100:.1f}%
                 k=10 {genus_knn[10]*100:.1f}%

  Genus Probe:   train={genus_probe['train_accuracy']*100:.1f}%  test={genus_probe['test_accuracy']*100:.1f}%

  Phylum k-NN:   k=1  {phylum_knn[1]*100:.1f}%
  Phylum Probe:  test={phylum_probe['test_accuracy']*100:.1f}%

  Wall time:     {time.time() - t0:.0f}s
{'='*60}""")


if __name__ == "__main__":
    main()
