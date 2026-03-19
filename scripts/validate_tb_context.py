#!/usr/bin/env python3
"""
Quick validation: does B-JEPA capture functionally relevant genomic context in TB?

Test: embed windows around known resistance loci vs random genomic regions
from H37Rv. If resistance loci cluster separately, B-JEPA encodes functional
information relevant to the diagnostic pipeline.

No experimental data needed — uses known biology as ground truth.

Usage:
    python scripts/validate_tb_context.py \
        --checkpoint outputs/checkpoints/v7.0/epoch0004.pt \
        --h37rv-fasta data/genomes/GCF_000195955.2_ASM19595v2_genomic.fna \
        --output-dir outputs/eval/tb_context_validation
"""
import argparse, os, sys, json, time
import numpy as np
import torch

# ── Known TB resistance loci (H37Rv coordinates) ───────────────────────────
# These are the genes/regions where resistance mutations occur
TB_RESISTANCE_LOCI = {
    # Gene: (start, end, drug, description)
    "rpoB": (759807, 763325, "Rifampicin", "RNA polymerase beta subunit"),
    "katG": (2153889, 2156111, "Isoniazid", "Catalase-peroxidase"),
    "inhA_promoter": (1674202, 1674502, "Isoniazid", "InhA promoter region"),
    "inhA": (1674202, 1675011, "Isoniazid/Ethionamide", "Enoyl-ACP reductase"),
    "embB": (4246514, 4249810, "Ethambutol", "Arabinosyltransferase B"),
    "embA": (4243147, 4246517, "Ethambutol", "Arabinosyltransferase A"),
    "pncA": (2288681, 2289241, "Pyrazinamide", "Pyrazinamidase"),
    "gyrA": (7302, 9818, "Fluoroquinolones", "DNA gyrase subunit A"),
    "gyrB": (5123, 7267, "Fluoroquinolones", "DNA gyrase subunit B"),
    "rrs": (1471846, 1473382, "Aminoglycosides", "16S rRNA"),
    "eis_promoter": (2714124, 2714424, "Kanamycin", "Eis promoter"),
    "ethA": (3986844, 3988386, "Ethionamide", "Monooxygenase"),
    "rpsL": (781560, 781934, "Streptomycin", "30S ribosomal protein S12"),
    "IS6110": (3121076, 3122421, "TB-specific", "Insertion sequence (species ID)"),
}

# Housekeeping genes (essential, highly conserved, NOT resistance-related)
TB_HOUSEKEEPING = {
    "dnaA": (1, 1524, "Replication initiation"),
    "dnaN": (2052, 3260, "DNA polymerase III beta"),
    "groEL2": (540786, 542402, "Chaperone"),
    "rpoA": (763370, 764326, "RNA polymerase alpha"),
    "sigA": (2518115, 2519365, "Principal sigma factor"),
    "ftsZ": (2410837, 2412032, "Cell division protein"),
    "murA": (3882043, 3883314, "Peptidoglycan biosynthesis"),
    "dnaG": (3098837, 3100582, "DNA primase"),
    "recA": (2846720, 2847784, "Recombinase A"),
    "gltA": (634988, 636349, "Citrate synthase"),
}


def load_genome(fasta_path):
    """Load H37Rv genome from FASTA."""
    sequence = []
    with open(fasta_path) as f:
        for line in f:
            if not line.startswith(">"):
                sequence.append(line.strip().upper())
    return "".join(sequence)


def extract_windows(genome, loci, window_size=2048):
    """Extract windows centered on each locus."""
    windows = []
    for name, info in loci.items():
        if len(info) == 4:
            start, end, drug, desc = info
        else:
            start, end, desc = info
            drug = None

        center = (start + end) // 2
        win_start = max(0, center - window_size // 2)
        win_end = win_start + window_size

        if win_end > len(genome):
            win_end = len(genome)
            win_start = win_end - window_size

        seq = genome[win_start:win_end]
        windows.append({
            "name": name,
            "sequence": seq,
            "start": win_start,
            "end": win_end,
            "type": "resistance" if drug else "housekeeping",
            "drug": drug,
            "description": desc if isinstance(desc, str) else info[-1],
        })
    return windows


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

    sd = ckpt["model_state_dict"]
    tgt = {k.replace("target_encoder.", ""): v for k, v in sd.items()
           if k.startswith("target_encoder.")}
    if tgt:
        encoder.load_state_dict(tgt)
    else:
        ctx = {k.replace("context_encoder.", ""): v for k, v in sd.items()
               if k.startswith("context_encoder.")}
        encoder.load_state_dict(ctx)

    return encoder.to(device).eval(), cfg


def embed_sequences(model, sequences, tokenizer_path, max_len, device):
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(tokenizer_path)

    embeddings = []
    for seq in sequences:
        ids = tok.encode(seq).ids[:max_len]
        pad = max_len - len(ids)
        tokens = torch.tensor(ids + [0] * pad).unsqueeze(0).to(device)
        pos = torch.arange(max_len, device=device).unsqueeze(0)

        with torch.no_grad():
            out = model(tokens, pos)

        cls = None
        if isinstance(out, dict):
            for k in ["cls", "context_cls"]:
                if k in out:
                    cls = out[k]
                    break
            if cls is None:
                cls = list(out.values())[0]
        else:
            cls = out

        if cls.dim() == 3:
            cls = cls[:, 0, :]
        embeddings.append(cls.squeeze().cpu().numpy())

    return np.array(embeddings)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--h37rv-fasta", required=True,
                        help="Path to H37Rv genome FASTA")
    parser.add_argument("--tokenizer-path", default="data/tokenizer/bpe_4096.json")
    parser.add_argument("--n-random", type=int, default=50,
                        help="Number of random genomic windows")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    device = torch.device(args.device)
    t0 = time.time()

    # Load genome
    print("[1/4] Loading H37Rv genome...")
    genome = load_genome(args.h37rv_fasta)
    print(f"  Genome length: {len(genome):,} bp")

    # Extract windows
    print("[2/4] Extracting genomic windows...")
    resistance_windows = extract_windows(genome, TB_RESISTANCE_LOCI)
    housekeeping_windows = extract_windows(genome, TB_HOUSEKEEPING)

    # Random windows (avoiding known loci)
    import random
    random.seed(42)
    known_ranges = set()
    for w in resistance_windows + housekeeping_windows:
        for p in range(w["start"], w["end"]):
            known_ranges.add(p)

    random_windows = []
    attempts = 0
    while len(random_windows) < args.n_random and attempts < 10000:
        pos = random.randint(0, len(genome) - 2048)
        if pos not in known_ranges:
            random_windows.append({
                "name": f"random_{len(random_windows)}",
                "sequence": genome[pos:pos + 2048],
                "start": pos,
                "end": pos + 2048,
                "type": "random",
                "drug": None,
                "description": "Random genomic region",
            })
        attempts += 1

    all_windows = resistance_windows + housekeeping_windows + random_windows
    print(f"  Resistance: {len(resistance_windows)}")
    print(f"  Housekeeping: {len(housekeeping_windows)}")
    print(f"  Random: {len(random_windows)}")

    # Embed
    print("[3/4] Extracting B-JEPA embeddings...")
    model, cfg = load_bjepa(args.checkpoint, device)
    max_len = cfg.get("max_seq_len", 512)

    sequences = [w["sequence"] for w in all_windows]
    embeddings = embed_sequences(model, sequences, args.tokenizer_path, max_len, device)
    print(f"  Embeddings: {embeddings.shape}")

    # Analyze
    print("[4/4] Analyzing...")
    from sklearn.preprocessing import normalize

    labels = [w["type"] for w in all_windows]
    names = [w["name"] for w in all_windows]

    # Pairwise cosine similarity
    X = normalize(embeddings, norm="l2")
    sim = X @ X.T

    # Within-group and between-group similarity
    types = list(set(labels))
    print(f"\n  Pairwise cosine similarity (mean):")
    for t1 in types:
        for t2 in types:
            mask1 = np.array([l == t1 for l in labels])
            mask2 = np.array([l == t2 for l in labels])
            sub_sim = sim[np.ix_(mask1, mask2)]
            if t1 == t2:
                # Exclude diagonal
                np.fill_diagonal(sub_sim, 0)
                n = sub_sim.shape[0]
                mean_sim = sub_sim.sum() / (n * (n - 1)) if n > 1 else 0
            else:
                mean_sim = sub_sim.mean()
            print(f"    {t1:<15} vs {t2:<15} = {mean_sim:.4f}")

    # Resistance loci: do same-drug targets cluster?
    print(f"\n  Same-drug target similarity:")
    res_windows = [w for w in all_windows if w["type"] == "resistance"]
    res_emb = embeddings[:len(resistance_windows)]
    res_X = normalize(res_emb, norm="l2")
    res_sim = res_X @ res_X.T

    for i, w1 in enumerate(res_windows):
        sims = []
        for j, w2 in enumerate(res_windows):
            if i != j:
                sims.append((w2["name"], res_sim[i, j]))
        sims.sort(key=lambda x: -x[1])
        top3 = sims[:3]
        print(f"    {w1['name']:<20} ({w1['drug']:<15}) → "
              f"nearest: {top3[0][0]} ({top3[0][1]:.3f}), "
              f"{top3[1][0]} ({top3[1][1]:.3f}), "
              f"{top3[2][0]} ({top3[2][1]:.3f})")

    # UMAP visualization
    print(f"\n  Generating UMAP...")
    try:
        from umap import UMAP
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        reducer = UMAP(n_components=2, n_neighbors=15, min_dist=0.1,
                       metric="cosine", random_state=42)
        umap_2d = reducer.fit_transform(embeddings)

        fig, ax = plt.subplots(figsize=(12, 8))

        colors = {"resistance": "#E24B4A", "housekeeping": "#378ADD",
                  "random": "#B4B2A9"}
        markers = {"resistance": "D", "housekeeping": "s", "random": "o"}

        for t in ["random", "housekeeping", "resistance"]:
            mask = [l == t for l in labels]
            idx = np.where(mask)[0]
            ax.scatter(umap_2d[idx, 0], umap_2d[idx, 1],
                       c=colors[t], marker=markers[t], s=60 if t != "random" else 20,
                       alpha=0.7 if t != "random" else 0.3,
                       label=f"{t} ({sum(mask)})", zorder=3 if t == "resistance" else 1)

        # Label resistance loci
        for i, w in enumerate(all_windows):
            if w["type"] == "resistance":
                ax.annotate(w["name"], (umap_2d[i, 0], umap_2d[i, 1]),
                            fontsize=7, alpha=0.8,
                            xytext=(5, 5), textcoords="offset points")

        ax.legend(loc="best", fontsize=10)
        ax.set_title("B-JEPA embeddings of TB genomic regions")
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")

        os.makedirs(args.output_dir, exist_ok=True)
        path = os.path.join(args.output_dir, "umap_tb_loci.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"    Saved: {path}")
    except Exception as e:
        print(f"    UMAP failed: {e}")

    # Save summary
    os.makedirs(args.output_dir, exist_ok=True)
    summary = {
        "checkpoint": args.checkpoint,
        "n_resistance": len(resistance_windows),
        "n_housekeeping": len(housekeeping_windows),
        "n_random": len(random_windows),
        "resistance_loci": [w["name"] for w in resistance_windows],
        "wall_time": time.time() - t0,
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Wall time: {time.time()-t0:.0f}s")
    print(f"  Output: {args.output_dir}")


if __name__ == "__main__":
    main()
