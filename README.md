# B-JEPA: Bacterial Joint-Embedding Predictive Architecture

<p align="center">
  <em>Self-supervised foundation model for bacterial genomics</em>
</p>

<p align="center">
  <a href="https://huggingface.co/orgava/dna-bacteria-jepa">🤗 Model</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#architecture">Architecture</a> •
  <a href="docs/">Docs</a>
</p>

---

<div align="center">

## Models

| | params | architecture | seq len | tokenizer | RankMe | weights |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **v3.1** | 8.5M | 6L x 384D x 6H | 512 | char-level | 372 / 384 | [checkpoint](https://huggingface.co/orgava/dna-bacteria-jepa) |
| **v4.0** | 48M | 12L x 576D x 9H | 1024 | BPE (4096) | -- | in progress |

</div>

## Architecture

<img width="2816" height="1317" alt="Gemini_Generated_Image_3kttjv3kttjv3ktt" src="https://github.com/user-attachments/assets/ce22938e-9695-41d2-ba3b-5f8a9f08a1bd" />

## Quick start

### Install
```bash
git clone https://github.com/VUzan-bio/bdna-jepa.git && cd bdna-jepa
pip install -e ".[all]"
```

### Pretrain
```bash
python scripts/pretrain.py --config configs/training/v4.0.yaml     # 48M model (A100)
python scripts/pretrain.py --config configs/training/v3.1.yaml     # 8.5M baseline (any GPU)
```

### Evaluate
```bash
python scripts/evaluate.py --checkpoint outputs/checkpoints/v4.0/best.pt --version v4.0
# → RankMe, kNN, linear probe, GC R², clustering, UMAP
```

### Use as feature extractor
```python
from bdna_jepa import load_encoder

encoder = load_encoder("orgava/dna-bacteria-jepa", version="v4.0")
embeddings = encoder.encode_sequences(["ATCGATCG..."])  # (1, 576)
```

## Training

B-JEPA optimizes four losses jointly, balanced by [GradNorm](https://arxiv.org/abs/1711.02257):

```
L_total = w_mlm · L_MLM  +  w_jepa · L_JEPA  +  w_var · L_var  +  w_cov · L_cov
```

| loss | what | why |
|---|---|---|
| L_MLM | cross-entropy on 15% masked tokens | token-level understanding |
| L_JEPA | MSE between predicted and target CLS | sequence-level functional context |
| L_var | hinge: per-dim std ≥ 1 ([VICReg](https://arxiv.org/abs/2105.04906)) | prevent complete collapse |
| L_cov | off-diagonal covariance penalty | prevent dimensional collapse |

Key training details: AdamW (lr=1e-3, cosine → 1e-6), bfloat16, batch 256, 300 epochs, EMA 0.996→1.0 cosine schedule. See [`configs/training/v4.0.yaml`](configs/training/v4.0.yaml) for full config.

## v3.1 → v4.0

| | v3.1 | v4.0 | why |
|---|---|---|---|
| scale | 8.5M | 48M | 5.6× for 417-species corpus |
| tokenizer | char-level | BPE (4096) | ~5× compression, learns motifs |
| positions | learned | RoPE | length generalization |
| predictor | 384D (same width) | 192D (0.33× bottleneck) | [I-JEPA](https://arxiv.org/abs/2301.08243): forces richer encoder |
| collapse fix | SIGReg | VICReg (var + cov) | SIGReg collapsed at epoch 80 |
| balancing | static weights | GradNorm (α=1.5) | JEPA gradient was 512× larger |
| JEPA target | token-block | CLS latent | [JEPA-DNA](https://arxiv.org/abs/2602.17162): 2D masking fails on 1D DNA |
| fragment JEPA | — | ✓ | cross-fragment genome context |

## SABER

**[SABER](https://github.com/VUzan-bio/saber)** (Systematic Automated Biosensor Engineering for Resistance) uses B-JEPA embeddings for CRISPR-Cas12a diagnostic design targeting multidrug-resistant tuberculosis.

```python
# in the SABER repo
from bdna_jepa import load_encoder
encoder = load_encoder("orgava/dna-bacteria-jepa", version="v4.0")
# → score crRNA candidates, design multiplex panels, predict mismatch tolerance
```

SABER pipeline: reference genome → WHO mutation catalogue → crRNA enumeration → **B-JEPA activity scoring** → multiplex optimization. Target: 14-plex electrochemical biosensor on laser-induced graphene. Part of a BRIDGE Discovery project at ETH Zurich.

## Repository

```
bdna-jepa/
├── bdna_jepa/                  # core library (pip install -e .)
│   ├── models/                 #   encoder, predictor, jepa
│   ├── losses/                 #   JEPA + MLM + VICReg + GradNorm
│   ├── data/                   #   tokenizer, dataset, masking
│   ├── training/               #   trainer
│   ├── evaluation/             #   probing, clustering, visualization
│   ├── utils/                  #   RankMe, features, logging
│   ├── config.py               #   versioned dataclass configs
│   └── hub.py                  #   HuggingFace load/save
├── configs/                    #   YAML: model, training, evaluation
├── scripts/                    #   pretrain · evaluate · visualize · export
├── tools/                      #   download_genomes · extract · tokenize
├── tests/
└── docs/
```

## References

- [I-JEPA](https://arxiv.org/abs/2301.08243) (Assran et al., 2023) — predictor bottleneck, EMA schedule
- [JEPA-DNA](https://arxiv.org/abs/2602.17162) (Larey et al., 2026) — dual MLM+CLS-JEPA on DNA
- [C-JEPA](https://arxiv.org/abs/2407.09394) (Mo et al., 2024) — VICReg for JEPA
- [V-JEPA](https://arxiv.org/abs/2404.08471) (Bardes et al., 2024) — multi-scale masking
- [ProkBERT](https://doi.org/10.1093/nar/gkae1070) (Ligeti et al., 2024) — prokaryote MLM baseline
- [GradNorm](https://arxiv.org/abs/1711.02257) (Chen et al., 2018) — multi-task gradient balancing
- [VICReg](https://arxiv.org/abs/2105.04906) (Bardes et al., 2022) — variance-covariance regularization

## Citation

```bibtex
@misc{uzan2026bjepa,
  title   = {B-JEPA: Self-Supervised Bacterial Genomic Foundation Model
             via Joint-Embedding Predictive Architectures},
  author  = {Uzan, Valentin},
  year    = {2026},
  url     = {https://github.com/VUzan-bio/bdna-jepa}
}
```

## License

MIT
