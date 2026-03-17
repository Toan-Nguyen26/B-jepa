#!/usr/bin/env python3
"""
Baselines for genus classification: TNF (tetranucleotide frequency) and random embeddings.
No pretrained model needed — pure feature engineering vs random control.

Usage:
    python scripts/benchmark_baselines.py \
        --data-path data/processed/eval_20k.csv \
        --taxonomy data/processed/genome_taxonomy.csv \
        --output-dir outputs/eval/baselines
"""
import argparse, csv, json, os, random, time, itertools
import numpy as np
from collections import Counter


def compute_tnf(sequence, k=4):
    """Compute tetranucleotide frequency vector (256-dim for k=4)."""
    bases = "ACGT"
    kmers = ["".join(c) for c in itertools.product(bases, repeat=k)]
    kmer_to_idx = {km: i for i, km in enumerate(kmers)}
    counts = np.zeros(len(kmers))
    seq = sequence.upper().replace("N", "")
    for i in range(len(seq) - k + 1):
        kmer = seq[i:i+k]
        if kmer in kmer_to_idx:
            counts[kmer_to_idx[kmer]] += 1
    total = counts.sum()
    if total > 0:
        counts /= total
    return counts


def knn_accuracy(embeddings, labels, k_values=[1, 5, 10, 20]):
    from sklearn.preprocessing import normalize
    X = normalize(embeddings, norm="l2")
    sim = X @ X.T
    np.fill_diagonal(sim, -1)
    results = {}
    for k in k_values:
        topk = np.argsort(-sim, axis=1)[:, :k]
        correct = sum(
            Counter(labels[topk[i]]).most_common(1)[0][0] == labels[i]
            for i in range(len(labels))
        )
        results[k] = correct / len(labels)
    return results


def linear_probe(embeddings, labels, min_per_class=5):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    counts = Counter(labels)
    valid = {c for c, n in counts.items() if n >= min_per_class}
    if len(valid) < 2:
        return {"test_accuracy": 0.0, "train_accuracy": 0.0, "n_classes": 0}

    mask = np.array([l in valid for l in labels])
    X, y = embeddings[mask], labels[mask]
    uniq = sorted(set(y))
    remap = {old: new for new, old in enumerate(uniq)}
    y = np.array([remap[l] for l in y])

    strat = y if int(len(y) * 0.2) >= len(uniq) else None
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=strat)
    sc = StandardScaler()
    X_tr, X_te = sc.fit_transform(X_tr), sc.transform(X_te)
    clf = LogisticRegression(max_iter=1000, solver="lbfgs", C=1.0, random_state=42)
    clf.fit(X_tr, y_tr)
    return {
        "train_accuracy": float(clf.score(X_tr, y_tr)),
        "test_accuracy": float(clf.score(X_te, y_te)),
        "n_classes": len(uniq),
    }


def compute_rankme(emb):
    U, S, _ = np.linalg.svd(emb - emb.mean(0), full_matrices=False)
    S = S / S.sum()
    S = S[S > 1e-12]
    return float(np.exp(-np.sum(S * np.log(S))))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--taxonomy", required=True)
    parser.add_argument("--n-samples", type=int, default=5000)
    parser.add_argument("--min-fragments", type=int, default=200)
    parser.add_argument("--max-per-genus", type=int, default=200)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    t0 = time.time()

    # Load data (same sampling as other benchmarks)
    print("[1/4] Loading data...")
    tax_map = {}
    with open(args.taxonomy) as f:
        for row in csv.DictReader(f):
            tax_map[row["genome"]] = {"genus": row["genus"], "phylum": row["phylum"]}

    genus_to_rows = {}
    with open(args.data_path) as f:
        for row in csv.DictReader(f):
            g = row["genome"]
            if g not in tax_map or tax_map[g]["genus"] == "Unknown":
                continue
            genus = tax_map[g]["genus"]
            genus_to_rows.setdefault(genus, []).append(row)

    eligible = {g: r for g, r in genus_to_rows.items() if len(r) >= args.min_fragments}
    print(f"  Genera >= {args.min_fragments}: {len(eligible)}")

    random.seed(42)
    samples = []
    per = min(args.max_per_genus, args.n_samples // max(len(eligible), 1))
    for genus, rows in sorted(eligible.items()):
        for row in random.sample(rows, min(per, len(rows))):
            samples.append({
                "seq": row["sequence"],
                "gc": float(row["gc_content"]),
                "genus": genus,
                "phylum": tax_map[row["genome"]]["phylum"],
            })

    genera = sorted(set(s["genus"] for s in samples))
    phyla = sorted(set(s["phylum"] for s in samples))
    g2id = {g: i for i, g in enumerate(genera)}
    p2id = {p: i for i, p in enumerate(phyla)}
    gids = np.array([g2id[s["genus"]] for s in samples])
    pids = np.array([p2id[s["phylum"]] for s in samples])
    gc = np.array([s["gc"] for s in samples])
    print(f"  {len(samples)} samples, {len(genera)} genera, {len(phyla)} phyla")

    results = {}

    # ── TNF Baseline ────────────────────────────────────────────────────
    print("\n[2/4] Computing TNF embeddings...")
    tnf_emb = np.array([compute_tnf(s["seq"]) for s in samples])
    print(f"  TNF: {tnf_emb.shape[0]} x {tnf_emb.shape[1]}")

    tnf_rankme = compute_rankme(tnf_emb)
    tnf_g_knn = knn_accuracy(tnf_emb, gids)
    tnf_p_knn = knn_accuracy(tnf_emb, pids, [1, 5, 10])
    tnf_g_probe = linear_probe(tnf_emb, gids)
    tnf_p_probe = linear_probe(tnf_emb, pids)

    from sklearn.linear_model import Ridge
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    Xtr, Xte, ytr, yte = train_test_split(tnf_emb, gc, test_size=0.2, random_state=42)
    sc = StandardScaler()
    tnf_gc_r2 = Ridge(1.0).fit(sc.fit_transform(Xtr), ytr).score(sc.transform(Xte), yte)

    results["tnf"] = {
        "method": "TNF (4-mer frequency)",
        "dim": 256, "params": 0,
        "rankme": tnf_rankme, "gc_r2": tnf_gc_r2,
        "genus_knn": {str(k): v for k, v in tnf_g_knn.items()},
        "phylum_knn": {str(k): v for k, v in tnf_p_knn.items()},
        "genus_probe": tnf_g_probe, "phylum_probe": tnf_p_probe,
    }

    # ── Random Embedding Baseline ───────────────────────────────────────
    print("\n[3/4] Computing random embeddings...")
    np.random.seed(42)
    rand_emb = np.random.randn(len(samples), 576).astype(np.float32)
    rand_rankme = compute_rankme(rand_emb)
    rand_g_knn = knn_accuracy(rand_emb, gids)
    rand_p_knn = knn_accuracy(rand_emb, pids, [1, 5, 10])
    rand_g_probe = linear_probe(rand_emb, gids)
    rand_p_probe = linear_probe(rand_emb, pids)

    results["random"] = {
        "method": "Random embeddings (576D Gaussian)",
        "dim": 576, "params": 0,
        "rankme": rand_rankme, "gc_r2": 0.0,
        "genus_knn": {str(k): v for k, v in rand_g_knn.items()},
        "phylum_knn": {str(k): v for k, v in rand_p_knn.items()},
        "genus_probe": rand_g_probe, "phylum_probe": rand_p_probe,
    }

    # ── GC-Only Baseline (1D feature) ──────────────────────────────────
    print("\n[3.5/4] Computing GC-only baseline...")
    gc_emb = gc.reshape(-1, 1)
    gc_g_knn = knn_accuracy(gc_emb, gids)
    gc_g_probe = linear_probe(gc_emb, gids)
    gc_p_probe = linear_probe(gc_emb, pids)

    results["gc_only"] = {
        "method": "GC content only (1D)",
        "dim": 1, "params": 0,
        "genus_knn": {str(k): v for k, v in gc_g_knn.items()},
        "genus_probe": gc_g_probe, "phylum_probe": gc_p_probe,
    }

    # ── Print Results ───────────────────────────────────────────────────
    wt = time.time() - t0
    print(f"""
{'='*70}
  Baseline Results
{'='*70}

  TNF (4-mer frequency, 256D, 0 params):
    RankMe:        {tnf_rankme:.1f}
    GC R2:         {tnf_gc_r2:.4f}
    Genus k-NN:    k=1  {tnf_g_knn[1]*100:.1f}%   k=5  {tnf_g_knn[5]*100:.1f}%   k=10  {tnf_g_knn[10]*100:.1f}%
    Genus Probe:   train={tnf_g_probe['train_accuracy']*100:.1f}%  test={tnf_g_probe['test_accuracy']*100:.1f}%
    Phylum Probe:  test={tnf_p_probe['test_accuracy']*100:.1f}%

  Random (576D Gaussian, 0 params):
    Genus k-NN:    k=1  {rand_g_knn[1]*100:.1f}%
    Genus Probe:   test={rand_g_probe['test_accuracy']*100:.1f}%

  GC-Only (1D):
    Genus k-NN:    k=1  {gc_g_knn[1]*100:.1f}%
    Genus Probe:   test={gc_g_probe['test_accuracy']*100:.1f}%

  Wall time:     {wt:.0f}s
{'='*70}""")

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: {args.output_dir}/summary.json")


if __name__ == "__main__":
    main()
