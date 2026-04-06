# B-JEPA

**Bacterial DNA Joint-Embedding Predictive Architecture**

Self-supervised foundation model for bacterial genomics. Pretrained on 6,326 complete bacterial reference genomes (~60M fragments) using a latent grounding framework: JEPA shapes the representation space, MLM drives token-level learning. Designed for downstream transfer to TB drug resistance prediction and CRISPR-Cas12a guide design.

[Model](https://huggingface.co/orgava/dna-bacteria-jepa) ·
[COMPASS Pipeline](https://github.com/VUzan-bio/compass) ·
[Architecture](#architecture) ·
[Quick Start](#quick-start) ·
[Citation](#citation)

---

## Overview

B-JEPA learns general-purpose bacterial DNA representations through a dual-objective self-supervised framework inspired by [I-JEPA](https://arxiv.org/abs/2301.08243) and [JEPA-DNA](https://arxiv.org/abs/2602.17162):

1. **Context encoder** processes partially-masked genomic sequences (only visible tokens, with original position IDs via RoPE)
2. **Cross-attention predictor** maps context token embeddings to target token embeddings in latent space
3. **EMA target encoder** provides stable prediction targets (cosine schedule τ: 0.996 → 1.0)
4. **MLM head** grounds representations at the token level within visible tokens
5. **SIGReg + per-dimension variance floor** prevents representational collapse
6. **GC adversarial** debiases embeddings from GC-content shortcuts

**v7.0** introduces dynamic loss scheduling — JEPA weight decays (5→1) while MLM weight ramps (1→5) over training, implementing the "latent grounding" principle from JEPA-DNA (Larey et al., 2026): JEPA shapes the representation space early, then MLM drives learning.

## Models

| Version | Params | Architecture | Data | RankMe | Status |
|---------|--------|-------------|------|--------|--------|
| v3.1 | 8.5M | 6L × 384D × 6H | 301K frags, char-level | 372/384 | [weights](https://huggingface.co/orgava/dna-bacteria-jepa) |
| v6.2 | 64.4M | 12L × 576D × 9H | 2M frags, BPE 4096 | 476/576 | checkpoint |
| **v7.0** | 64.4M | 12L × 576D × 9H | **10M frags** (from 50M pool), BPE 4096 | — | **training** |

## Architecture

### Loss Function

```
L_total = w_jepa(t) · L_JEPA + w_mlm(t) · L_MLM + w_sig · L_SIGReg + w_gc · L_GC_adv

where (v7.0 dynamic schedule):
  w_jepa(t) = 5.0 · (1 - 0.8t)     # 5.0 → 1.0 (cosine)
  w_mlm(t)  = 1.0 + 4.0t            # 1.0 → 5.0 (cosine)
  t = 0.5 · (1 - cos(π · epoch/N))  # progress 0→1
```

| Loss | Formulation | Role |
|------|-------------|------|
| L_JEPA | Smooth L1 on per-token predictions at masked positions | Sequence-level context (latent grounding) |
| L_MLM | Cross-entropy on 15% masked visible tokens | Token-level sequence understanding (primary) |
| L_SIGReg | Epps-Pulley Gaussianity test + per-dim variance floor (γ=1.0) | Collapse prevention |
| L_GC_adv | MSE with gradient reversal (Ganin schedule) | GC-content debiasing |

### Components

**Context Encoder.** 12-layer Transformer with BPE tokenization (vocab 4096), RoPE positional embeddings, RMSNorm, SwiGLU feedforward, QK-Norm. Processes only visible tokens — never sees masked positions. Prepends learnable [CLS] token.

**JEPA Predictor.** 6-layer cross-attention Transformer (384D, 6 heads). Learnable mask tokens at target positions cross-attend to context embeddings, then self-attend among themselves. Projects predictions back to encoder dimension (576D). Follows the I-JEPA narrow bottleneck principle (0.67× encoder width).

**EMA Target Encoder.** Structural copy of context encoder (dropout=0). Updated via exponential moving average with cosine schedule (τ: 0.996 → 1.0). Receives the full unmasked sequence to provide stable prediction targets.

**Multi-Block Masking.** 4 contiguous target blocks with curriculum scheduling: mask ratio 50%→70%, minimum block length 10→30 tokens over training. Adapted from I-JEPA's spatial masking to 1D genomic sequences.

### Design Evolution

| | v3.1 | v6.2 | v7.0 | Rationale |
|---|------|------|------|-----------|
| Data | 301K frags | 2M frags | **10M frags** (50M pool) | SSL needs data diversity |
| Tokenizer | Char-level | BPE 4096 | BPE 4096 | ~5× compression; learns motifs |
| Positional | Learned | RoPE | RoPE | Length generalization |
| JEPA type | CLS-to-CLS | Token-level cross-attn | Token-level cross-attn | Per-position prediction |
| Collapse prevention | SIGReg | SIGReg + var floor | SIGReg + per-dim var floor | Prevents norm and dimensional collapse |
| Loss schedule | Static | Static | **Dynamic JEPA→MLM** | Latent grounding (JEPA-DNA) |
| GC debiasing | — | GC adversarial | GC adversarial | Prevents GC-content shortcuts |
| Validation | — | — | **5% genome-level holdout** | Honest overfitting detection |

## Quick Start

### Installation

```bash
git clone https://github.com/VUzan-bio/bdna-jepa.git && cd bdna-jepa
pip install -e ".[all]"
```

### Pretraining

```bash
# v7.0 — 114.5M model with latent grounding (A100 recommended)
python scripts/fragment_genomes.py \
    --input data/genomes/ \
    --output data/processed/pretrain_full.csv \
    --window 2048 --stride 512

python bdna_jepa/models/jepa_v6/pretrain_v6.py \
    --data-path data/processed/pretrain_10M.csv \
    --tokenizer-path data/tokenizer/bpe_4096.json \
    --run-version v7.0 \
    --epochs 30 \
    --batch-size 128 \
    --lr 3e-4 \
    --warmup-epochs 1 \
    --dynamic-weights \
    --jepa-weight 5.0 \
    --mlm-weight-start 1.0 \
    --mlm-weight-end 5.0 \
    --predictor-dim 384 \
    --predictor-depth 6 \
    --var-gamma 1.0 \
    --val-frac 0.05 \
    --no-compile \
    --save-every 2
```

### Evaluation

```bash
python scripts/evaluate_v6.py \
    --checkpoint outputs/checkpoints/v7.0/epoch0005.pt \
    --species-map data/processed/genome_species.csv \
    --n-samples 20000
# → RankMe, species k-NN, linear probe, UMAP, SVD spectrum
```

### Feature Extraction

```python
import torch
from bdna_jepa.models.jepa_v6.pretrain_v6 import BJEPAv6

# Load checkpoint (config stored inside)
ckpt = torch.load("outputs/checkpoints/v7.0/epoch0005.pt", weights_only=False)
model = BJEPAv6(**{k: ckpt["config"][k] for k in [
    "embed_dim", "num_layers", "num_heads", "ff_dim", "max_seq_len",
    "predictor_dim", "predictor_depth", "predictor_heads", "var_gamma"
]}, vocab_size=4096)
model.load_state_dict(ckpt["model_state_dict"], strict=False)
model.eval().cuda()

# Extract CLS embeddings (uses EMA target encoder)
tokens = torch.randint(0, 4096, (1, 512)).cuda()
cls_embedding = model.encode(tokens, use_target=True)  # → (1, 576)
```

## Training Details

**Data.** 6,326 complete bacterial reference genomes downloaded from NCBI RefSeq via `ncbi-genome-download`. Fragmented into ~50M overlapping windows (2048bp, 512bp stride). 10M randomly sampled for v7.0 training (full 50M for scaling experiments). BPE tokenization (vocab 4096) compresses each fragment to ~512 tokens.

**Optimizer.** AdamW (peak LR 3e-4, cosine decay → 1e-6, warmup 1 epoch, weight decay 0.05, batch size 128). bfloat16 mixed precision. Gradient clipping at 1.0. 30 epochs on A100-40GB (~11.4h/epoch).

**Monitoring.** Training health tracked via: RankMe (effective embedding rank from SVD), per-dimension CLS std, embedding norms, JEPA cosine similarity, variance floor activation, GC correlation coefficient. UMAP/t-SNE/SVD visualizations generated every epoch. All metrics logged to Weights & Biases.

## Downstream: COMPASS

[**COMPASS**](https://compass-crispr.com/)

```
WHO mutation catalogue → target resolution (M1) → PAM scanning (M2)
→ candidate filtering (M3) → off-target screening (M4)
→ B-JEPA activity scoring (M5) → mismatch discrimination (M5.5)
→ synthetic enhancement (M6) → multiplex optimization (M7)
→ RPA primer co-design (M8) → panel assembly (M9)
```

## Repository Structure

```
bdna-jepa/
├── bdna_jepa/
│   ├── models/
│   │   └── jepa_v6/             # v6.2/v7.0 architecture + training
│   │       └── pretrain_v6.py   # BJEPAv6, ContextEncoder, JEPAPredictor, SIGReg
│   ├── data/                    # BPE tokenizer, dataset, masking
│   ├── evaluation/              # k-NN, linear probe, clustering, UMAP
│   └── hub.py                   # HuggingFace load/save
├── scripts/
│   ├── fragment_genomes.py      # Genome → fragment CSV generation
│   ├── evaluate_v6.py           # Post-training eval with species mapping
│   └── evaluate.py              # Legacy v4.0 evaluation
├── configs/                     # YAML configs (v3.1, v4.0)
├── data/
│   ├── genomes/                 # Raw .fna files (6,326 genomes)
│   ├── processed/               # Fragment CSVs + species mapping
│   └── tokenizer/               # BPE tokenizer JSON
└── outputs/
    ├── checkpoints/             # Model weights per version
    └── eval/                    # Evaluation results + figures
```

## References

- **I-JEPA** — Assran et al. (2023). [Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture.](https://arxiv.org/abs/2301.08243) *CVPR 2023.* Core architecture: predictor bottleneck, EMA schedule, multi-block masking.
- **JEPA-DNA** — Larey et al. (2026). [Grounding Genomic Foundation Models through Joint-Embedding Predictive Architectures.](https://arxiv.org/abs/2602.17162) Latent grounding: CLS-level JEPA + token-level MLM. Dynamic loss scheduling.
- **GeneJEPA** — Litman et al. (2025). [A Predictive World Model of the Transcriptome.](https://www.biorxiv.org/content/10.1101/2025.10.14.682378v1) Perceiver JEPA for single-cell RNA-seq on Tahoe-100M.
- **V-JEPA 2** — Bardes et al. (2025). [Self-Supervised Video Models Enable Understanding, Prediction and Planning.](https://arxiv.org/abs/2506.09985) Multi-scale masking, variance regularization.
- **SIGReg** — Garrido et al. (2024). Sketched Isotropic Gaussian Regularization via Epps-Pulley characteristic function test.
- **ProkBERT** — Ligeti et al. (2024). [Genomic Language Models for Microbiome Analysis.](https://doi.org/10.1093/nargab/lqae065) Prokaryote MLM baseline.
- **VICReg** — Bardes et al. (2022). [Variance-Invariance-Covariance Regularization for Self-Supervised Learning.](https://arxiv.org/abs/2105.04906) *ICLR 2022.*

## Citation

```bibtex
@misc{uzan2026bjepa,
    title   = {{B-JEPA}: Self-Supervised Bacterial Genomic Foundation Model
               via Joint-Embedding Predictive Architectures with Latent Grounding},
    author  = {Uzan, Valentin},
    year    = {2026},
    url     = {https://github.com/VUzan-bio/bdna-jepa}
}
```

## License

MIT
