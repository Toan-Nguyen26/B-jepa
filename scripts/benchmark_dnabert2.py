#!/usr/bin/env python3
"""DNABERT-2 benchmark with standard attention (no flash/triton)."""
import torch, os, sys, importlib, math, json, time, csv, argparse, random
import torch.nn.functional as F
import numpy as np
from collections import Counter


def load_dnabert2(device="cpu"):
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer, BertConfig

    model_name = "zhihan1996/DNABERT-2-117M"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    path = snapshot_download(model_name)

    # Make importable
    init_path = os.path.join(path, "__init__.py")
    if not os.path.exists(init_path):
        open(init_path, "w").close()
    parent, pkg = os.path.dirname(path), os.path.basename(path)
    if parent not in sys.path:
        sys.path.insert(0, parent)

    mod = importlib.import_module(pkg + ".bert_layers")
    config = BertConfig.from_pretrained(path)
    config.pad_token_id = tokenizer.pad_token_id or 0

    model = mod.BertModel(config)
    state = torch.load(os.path.join(path, "pytorch_model.bin"), map_location="cpu")
    cleaned = {k.replace("bert.", ""): v for k, v in state.items() if not k.startswith("cls.")}
    model.load_state_dict(cleaned, strict=False)

    num_heads = config.num_attention_heads
    head_dim = config.hidden_size // num_heads
    dim = config.hidden_size

    # Build ALiBi slopes
    def get_alibi_slopes(n):
        def power_of_2(n):
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            return [start * start ** i for i in range(n)]
        if math.log2(n).is_integer():
            return power_of_2(n)
        closest = 2 ** math.floor(math.log2(n))
        return power_of_2(closest) + get_alibi_slopes(2 * closest)[0::2][:n - closest]

    slopes = torch.tensor(get_alibi_slopes(num_heads)).float()

    # Replace forward entirely
    import types

    def safe_forward(self, input_ids, attention_mask=None, token_type_ids=None, **kwargs):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        emb = self.embeddings(input_ids, token_type_ids=token_type_ids)
        bs, sl, _ = emb.shape

        # ALiBi bias
        pos = torch.arange(sl, device=emb.device).float()
        rel = pos.unsqueeze(0) - pos.unsqueeze(1)
        alibi = slopes.to(emb.device).unsqueeze(1).unsqueeze(1) * rel.unsqueeze(0)
        alibi = alibi.unsqueeze(0).to(emb.dtype)

        # Padding mask
        ext_mask = (1.0 - attention_mask[:, None, None, :].float()) * -10000.0

        hidden = emb
        for layer in self.encoder.layer:
            # Self-attention with standard matmul
            qkv = layer.attention.self.Wqkv(hidden)
            qkv = qkv.reshape(bs, sl, 3, num_heads, head_dim)
            q, k, v = qkv.unbind(dim=2)
            q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
            scores = scores + alibi
            scores = scores + ext_mask
            attn = F.softmax(scores, dim=-1)
            ctx = torch.matmul(attn, v).transpose(1, 2).reshape(bs, sl, dim)

            # Attention output + residual + LN
            attn_out = layer.attention.output.dense(ctx)
            hidden = layer.attention.output.LayerNorm(attn_out + hidden)

            # MLP: gated linear unit
            gated = layer.mlp.gated_layers(hidden)
            h1, gate = gated.chunk(2, dim=-1)
            mlp_act = h1 * F.gelu(gate)
            mlp_out = layer.mlp.wo(mlp_act)
            hidden = layer.mlp.layernorm(mlp_out + hidden)

        class Out:
            pass
        r = Out()
        r.last_hidden_state = hidden
        return r

    model.forward = types.MethodType(safe_forward, model)
    model = model.to(device).eval()
    return model, tokenizer, config


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
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=strat)
    sc = StandardScaler()
    X_tr, X_te = sc.fit_transform(X_tr), sc.transform(X_te)
    clf = LogisticRegression(max_iter=1000, solver="lbfgs", C=1.0, random_state=42)
    clf.fit(X_tr, y_tr)
    return {
        "train_accuracy": float(clf.score(X_tr, y_tr)),
        "test_accuracy": float(clf.score(X_te, y_te)),
        "n_classes": len(uniq),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--taxonomy", required=True)
    parser.add_argument("--n-samples", type=int, default=5000)
    parser.add_argument("--min-fragments", type=int, default=200)
    parser.add_argument("--max-per-genus", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    device = torch.device(args.device)
    t0 = time.time()

    print("[1/5] Loading DNABERT-2...")
    model, tokenizer, config = load_dnabert2(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Params: {n_params:,}, Dim: {config.hidden_size}")

    print("[2/5] Loading data...")
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
                "seq": row["sequence"][:2048],
                "gc": float(row["gc_content"]),
                "genus": genus,
                "phylum": tax_map[row["genome"]]["phylum"],
            })

    genera = sorted(set(s["genus"] for s in samples))
    phyla = sorted(set(s["phylum"] for s in samples))
    g2id = {g: i for i, g in enumerate(genera)}
    p2id = {p: i for i, p in enumerate(phyla)}
    print(f"  {len(samples)} samples, {len(genera)} genera, {len(phyla)} phyla")

    print("[3/5] Extracting embeddings...")
    all_emb, all_gc, all_g, all_p = [], [], [], []
    for i in range(0, len(samples), args.batch_size):
        batch = samples[i:i + args.batch_size]
        seqs = [s["seq"] for s in batch]
        enc = tokenizer(seqs, return_tensors="pt", padding=True,
                        truncation=True, max_length=512).to(device)
        with torch.no_grad():
            out = model(**enc)
        h = out.last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1).float()
        pooled = (h * mask).sum(1) / mask.sum(1)
        all_emb.append(pooled.cpu().numpy())
        for s in batch:
            all_gc.append(s["gc"])
            all_g.append(g2id[s["genus"]])
            all_p.append(p2id[s["phylum"]])
        done = i + len(batch)
        if done % 200 < args.batch_size:
            print(f"    {done}/{len(samples)}")

    emb = np.concatenate(all_emb)
    gc = np.array(all_gc)
    gids = np.array(all_g)
    pids = np.array(all_p)
    print(f"  Done: {emb.shape[0]} x {emb.shape[1]}")

    print("[4/5] Computing metrics...")
    U, S, _ = np.linalg.svd(emb - emb.mean(0), full_matrices=False)
    S = S / S.sum()
    S = S[S > 1e-12]
    rankme = float(np.exp(-np.sum(S * np.log(S))))

    g_knn = knn_accuracy(emb, gids)
    p_knn = knn_accuracy(emb, pids, [1, 5, 10])
    g_probe = linear_probe(emb, gids)
    p_probe = linear_probe(emb, pids)

    from sklearn.linear_model import Ridge
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    Xtr, Xte, ytr, yte = train_test_split(emb, gc, test_size=0.2, random_state=42)
    sc = StandardScaler()
    gc_r2 = Ridge(1.0).fit(sc.fit_transform(Xtr), ytr).score(sc.transform(Xte), yte)

    wt = time.time() - t0
    print(f"""
{'='*60}
  DNABERT-2 Baseline
{'='*60}
  Params:        {n_params:,}
  RankMe:        {rankme:.1f}
  GC R2:         {gc_r2:.4f}

  Genus k-NN:    k=1  {g_knn[1]*100:.1f}%
                 k=5  {g_knn[5]*100:.1f}%
                 k=10 {g_knn[10]*100:.1f}%
                 k=20 {g_knn[20]*100:.1f}%

  Genus Probe:   train={g_probe['train_accuracy']*100:.1f}%  test={g_probe['test_accuracy']*100:.1f}%

  Phylum k-NN:   k=1  {p_knn[1]*100:.1f}%
  Phylum Probe:  test={p_probe['test_accuracy']*100:.1f}%

  Wall time:     {wt:.0f}s
{'='*60}""")

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump({
            "model": "zhihan1996/DNABERT-2-117M",
            "n_params": n_params,
            "embed_dim": config.hidden_size,
            "n_samples": len(emb),
            "rankme": rankme,
            "gc_r2": gc_r2,
            "genus_knn": {str(k): v for k, v in g_knn.items()},
            "phylum_knn": {str(k): v for k, v in p_knn.items()},
            "genus_probe": g_probe,
            "phylum_probe": p_probe,
            "genera": genera,
            "wall_time": wt,
        }, f, indent=2)
    print(f"  Saved: {args.output_dir}/summary.json")


if __name__ == "__main__":
    main()
