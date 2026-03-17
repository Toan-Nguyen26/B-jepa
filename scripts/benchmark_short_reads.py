#!/usr/bin/env python3
"""
Fragment length ablation: how do methods degrade as fragments get shorter?

At 2048bp, TNF has enough signal to match learned models.
At 150bp (real Illumina read length), k-mer statistics are noisy.
This is where foundation models should show their value.

Usage:
    python scripts/benchmark_short_reads.py \
        --checkpoint outputs/checkpoints/v7.0/epoch0004.pt \
        --data-path data/processed/eval_20k.csv \
        --taxonomy data/processed/genome_taxonomy.csv \
        --output-dir outputs/eval/short_reads
"""
import argparse, csv, json, os, sys, random, time, itertools
import numpy as np
import torch
from collections import Counter


# ── TNF ─────────────────────────────────────────────────────────────────────
def compute_tnf(sequence, k=4):
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


# ── Metrics ─────────────────────────────────────────────────────────────────
def knn_accuracy(emb, labels, k=5):
    from sklearn.preprocessing import normalize
    X = normalize(emb, norm="l2")
    sim = X @ X.T
    np.fill_diagonal(sim, -1)
    topk = np.argsort(-sim, axis=1)[:, :k]
    correct = sum(
        Counter(labels[topk[i]]).most_common(1)[0][0] == labels[i]
        for i in range(len(labels))
    )
    return correct / len(labels)


def linear_probe(emb, labels):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    counts = Counter(labels)
    valid = {c for c, n in counts.items() if n >= 5}
    if len(valid) < 2:
        return 0.0

    mask = np.array([l in valid for l in labels])
    X, y = emb[mask], labels[mask]
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
    return float(clf.score(X_te, y_te))


# ── B-JEPA embedding extraction ────────────────────────────────────────────
def load_bjepa(checkpoint_path, device):
    sys.path.insert(0, os.getcwd())
    from bdna_jepa.models.jepa_v6.pretrain_v6 import ContextEncoder

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})

    encoder = ContextEncoder(
        vocab_size=cfg.get("vocab_size", 4096),
        embed_dim=cfg.get("embed_dim", 576),
        num_layers=cfg.get("num_layers", 12),
        num_heads=cfg.get("num_heads", 9),
        ff_dim=cfg.get("ff_dim", 2304),
        max_seq_len=cfg.get("max_seq_len", 512),
    )

    # Load target encoder weights (EMA)
    def strip_prefix(sd, prefix):
        return {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}

    for key in ["model_state_dict", "target_encoder_state_dict"]:
        if key in ckpt:
            sd = ckpt[key]
            target_sd = strip_prefix(sd, "target_encoder.")
            if target_sd and len(target_sd) > 10:
                encoder.load_state_dict(target_sd)
                break
            context_sd = strip_prefix(sd, "context_encoder.")
            if context_sd and len(context_sd) > 10:
                encoder.load_state_dict(context_sd)
                break

    return encoder.to(device).eval(), cfg


def extract_bjepa_embeddings(model, sequences, tokenizer_path, max_len, device):
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(tokenizer_path)

    embeddings = []
    for i, seq in enumerate(sequences):
        enc = tok.encode(seq)
        ids = enc.ids[:max_len]
        pad_len = max_len - len(ids)
        tokens = torch.tensor(ids + [0] * pad_len, dtype=torch.long).unsqueeze(0).to(device)
        pos = torch.arange(max_len, device=device).unsqueeze(0)

        with torch.no_grad():
            out = model(tokens, pos)

        if isinstance(out, dict):
            for k in ["cls", "context_cls", "last_hidden_state"]:
                if k in out:
                    cls = out[k]
                    if cls.dim() == 3:
                        cls = cls[:, 0, :]
                    break
        else:
            cls = out[:, 0, :] if out.dim() == 3 else out

        embeddings.append(cls.float().cpu().numpy().squeeze())

        if (i + 1) % 200 == 0:
            print(f"    B-JEPA: {i+1}/{len(sequences)}")

    return np.array(embeddings)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--taxonomy", required=True)
    parser.add_argument("--tokenizer-path", default="data/tokenizer/bpe_4096.json")
    parser.add_argument("--n-samples", type=int, default=5000)
    parser.add_argument("--min-fragments", type=int, default=200)
    parser.add_argument("--max-per-genus", type=int, default=200)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    device = torch.device(args.device)
    t0 = time.time()

    # Load data
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

    random.seed(42)
    samples = []
    per = min(args.max_per_genus, args.n_samples // max(len(eligible), 1))
    for genus, rows in sorted(eligible.items()):
        for row in random.sample(rows, min(per, len(rows))):
            samples.append({
                "seq": row["sequence"],
                "genus": genus,
            })

    genera = sorted(set(s["genus"] for s in samples))
    g2id = {g: i for i, g in enumerate(genera)}
    gids = np.array([g2id[s["genus"]] for s in samples])
    full_seqs = [s["seq"] for s in samples]
    print(f"  {len(samples)} samples, {len(genera)} genera")

    # Load B-JEPA
    print("\n[2/4] Loading B-JEPA...")
    model, cfg = load_bjepa(args.checkpoint, device)
    max_len = cfg.get("max_seq_len", 512)

    # Test at different fragment lengths
    fragment_lengths = [150, 300, 500, 1000, 2048]
    results = {}

    for frag_len in fragment_lengths:
        print(f"\n{'='*60}")
        print(f"  Fragment length: {frag_len}bp")
        print(f"{'='*60}")

        # Truncate sequences to fragment length
        truncated = [seq[:frag_len] for seq in full_seqs]

        # TNF
        print(f"  Computing TNF...")
        tnf_emb = np.array([compute_tnf(seq) for seq in truncated])
        tnf_knn = knn_accuracy(tnf_emb, gids, k=5)
        tnf_probe = linear_probe(tnf_emb, gids)

        # B-JEPA
        print(f"  Computing B-JEPA embeddings...")
        bjepa_emb = extract_bjepa_embeddings(
            model, truncated, args.tokenizer_path, max_len, device)
        bjepa_knn = knn_accuracy(bjepa_emb, gids, k=5)
        bjepa_probe = linear_probe(bjepa_emb, gids)

        results[frag_len] = {
            "tnf_knn": tnf_knn,
            "tnf_probe": tnf_probe,
            "bjepa_knn": bjepa_knn,
            "bjepa_probe": bjepa_probe,
        }

        print(f"  TNF:    k-NN={tnf_knn*100:.1f}%  probe={tnf_probe*100:.1f}%")
        print(f"  B-JEPA: k-NN={bjepa_knn*100:.1f}%  probe={bjepa_probe*100:.1f}%")
        delta = bjepa_probe - tnf_probe
        winner = "B-JEPA" if delta > 0 else "TNF"
        print(f"  Winner: {winner} ({delta*100:+.1f}%)")

    # Summary table
    print(f"\n{'='*70}")
    print(f"  Fragment Length Ablation — Genus Classification (probe)")
    print(f"{'='*70}")
    print(f"  {'Length':<10} {'TNF':<12} {'B-JEPA':<12} {'Delta':<12} {'Winner'}")
    print(f"  {'-'*58}")
    for fl in fragment_lengths:
        r = results[fl]
        delta = r["bjepa_probe"] - r["tnf_probe"]
        winner = "B-JEPA" if delta > 0 else "TNF"
        print(f"  {fl:<10} {r['tnf_probe']*100:<12.1f} {r['bjepa_probe']*100:<12.1f} {delta*100:<+12.1f} {winner}")
    print(f"{'='*70}")

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"  Saved: {args.output_dir}/summary.json")
    print(f"  Wall time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
