"""
B-JEPA v4.5 — Bacterial DNA Foundation Model (Single-File Pretrain)
====================================================================

Architecture: I-JEPA per-token prediction + SIGReg + curriculum masking.

v4.5 fixes from v4.4 (collapsed at RankMe=32):
  - Per-token prediction at masked positions (v3 I-JEPA architecture)
    replaces CLS→CLS prediction which trivially collapses.
  - SIGReg (LeJEPA, Balestriero & LeCun 2025) replaces weak VICReg.
    One hyperparameter vs three, provably optimal distribution.
  - Multi-block masking (4 target blocks) instead of span MLM masking.
  - Curriculum masking: ratio 0.15→0.50, block length 3→15.
  - MSE prediction loss (not cosine — cosine hid the collapse).
  - No CLS token — mean-pool visible tokens (simpler, proven in v3).
  - UMAP colored by GC content. t-SNE uses max_iter (not n_iter).

Data pipeline:  BPE tokenizer (vocab=4096) + pretrain_2M.csv.
Encoder:        12L, 576D, 9H, ff=2304, pre-norm, GELU, learned pos.
Predictor:      4L, 384D, 6H, mask_token + learned pos_embed.
Loss:           MSE(pred, target) at masked positions + λ·SIGReg
                + seq_pred + RC + GC_adversary.
EMA:            Cosine schedule 0.996 → 1.0.
Training:       30 epochs, peak_lr=3e-4, cosine schedule, batch=64.

References:
  [1] Assran et al. "I-JEPA." CVPR 2023.
  [2] Mo et al. "C-JEPA." NeurIPS 2024.
  [3] Balestriero & LeCun. "LeJEPA." arXiv:2511.08544, 2025.
  [4] Schiff et al. "Caduceus." ICML 2024. (RC consistency)

Usage:
  cd /workspace/bdna-jepa
  python pretrain_v45.py --data-path data/processed/pretrain_2M.csv \
      --tokenizer-path data/tokenizer/bpe_4096.json \
      --epochs 30 --batch-size 64 --lr 3e-4
"""

from __future__ import annotations

import argparse
import copy
import math
import os
import random
import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Configuration
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class MaskingConfig:
    """I-JEPA multi-block masking with curriculum."""
    mask_ratio_start: float = 0.15
    mask_ratio_end: float = 0.50
    num_target_blocks: int = 4
    min_block_len_start: int = 3
    min_block_len_end: int = 15
    context_ratio_floor: float = 0.30
    min_mask_ratio: float = 0.10
    max_mask_ratio: float = 0.60

@dataclass
class JEPAConfig:
    """Top-level JEPA configuration."""
    predictor_dim: int = 384
    predictor_depth: int = 4
    predictor_num_heads: int = 6
    max_seq_len: int = 512
    ema_decay_start: float = 0.996
    ema_decay_end: float = 1.0
    masking: MaskingConfig = field(default_factory=MaskingConfig)
    sigreg_num_slices: int = 512
    sigreg_num_points: int = 17


# ═══════════════════════════════════════════════════════════════════════════════
# 2. SIGReg — Sketched Isotropic Gaussian Regularization
# ═══════════════════════════════════════════════════════════════════════════════

class SIGRegVectorized(nn.Module):
    """Vectorized SIGReg from LeJEPA (Balestriero & LeCun, 2025).

    Projects embeddings onto K random 1D directions, tests each marginal
    against N(0,1) via Epps-Pulley CF test. Loss = 0 when perfectly Gaussian.
    """

    def __init__(self, num_slices: int = 512, num_points: int = 17):
        super().__init__()
        self.num_slices = num_slices
        t_max = 2.0
        t_points = torch.linspace(0, t_max, num_points + 1)[1:]
        self.register_buffer('t_points', t_points)
        dt = t_max / num_points
        weights = torch.full((num_points,), dt)
        weights[0] = dt / 2
        weights[-1] = dt / 2
        self.register_buffer('weights', weights)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        B, D = embeddings.shape
        if B < 4:
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)

        directions = torch.randn(D, self.num_slices, device=embeddings.device, dtype=embeddings.dtype)
        directions = F.normalize(directions, dim=0)
        proj = embeddings @ directions  # (B, K)
        proj = (proj - proj.mean(dim=0, keepdim=True)) / (proj.std(dim=0, keepdim=True) + 1e-8)

        total = torch.tensor(0.0, device=embeddings.device)
        for t_idx, t in enumerate(self.t_points):
            cos_tp = torch.cos(t * proj)
            sin_tp = torch.sin(t * proj)
            ecf_real = cos_tp.mean(dim=0)
            ecf_imag = sin_tp.mean(dim=0)
            ecf_sq = ecf_real ** 2 + ecf_imag ** 2
            tcf = math.exp(-0.5 * t.item() ** 2)
            integrand = ecf_sq - 2 * ecf_real * tcf + tcf ** 2
            total = total + self.weights[t_idx] * integrand.mean()
        return total

    @torch.no_grad()
    def gaussianity_score(self, embeddings: torch.Tensor) -> float:
        return self.forward(embeddings.float()).item()


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Encoder — Pre-norm Transformer (no CLS token)
# ═══════════════════════════════════════════════════════════════════════════════

class DNATransformerEncoder(nn.Module):
    """Transformer encoder for DNA BPE tokens.

    No CLS token — mean-pool visible tokens for pooled representation.
    Uses nn.TransformerEncoderLayer with pre-norm (norm_first=True).
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 576,
        num_layers: int = 12,
        num_heads: int = 9,
        ff_dim: int = 2304,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        pad_token_id: int = 0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        self.pad_token_id = pad_token_id

        self.token_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_token_id)
        self.pos_embedding = nn.Embedding(max_seq_len, embed_dim)
        self.embed_dropout = nn.Dropout(dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=ff_dim, dropout=dropout,
            batch_first=True, norm_first=True,
            activation='gelu',
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.final_norm = nn.LayerNorm(embed_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.pos_embedding.weight, std=0.02)
        if self.token_embedding.padding_idx is not None:
            self.token_embedding.weight.data[self.token_embedding.padding_idx].zero_()

    def get_attention_mask(self, tokens: torch.Tensor) -> torch.Tensor:
        return tokens != self.pad_token_id

    def forward(
        self,
        tokens: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            pooled: (B, D) mean-pooled over valid positions
            token_embeddings: (B, L, D) per-token representations
        """
        if tokens.ndim == 1:
            tokens = tokens.unsqueeze(0)
        B, L = tokens.shape
        if attention_mask is None:
            attention_mask = self.get_attention_mask(tokens)

        positions = torch.arange(L, device=tokens.device).unsqueeze(0).expand(B, -1)
        x = self.token_embedding(tokens) + self.pos_embedding(positions)
        x = self.embed_dropout(x)

        key_padding_mask = ~attention_mask.bool()
        token_embeddings = self.encoder(x, src_key_padding_mask=key_padding_mask)
        token_embeddings = self.final_norm(token_embeddings)

        mask = attention_mask.unsqueeze(-1).float()
        pooled = (token_embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

        return pooled, token_embeddings


# ═══════════════════════════════════════════════════════════════════════════════
# 4. JEPAPredictor — I-JEPA Transformer with mask tokens
# ═══════════════════════════════════════════════════════════════════════════════

class _PredictorBlock(nn.Module):
    """Pre-norm transformer block for predictor."""
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 2.0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, dim), nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + h
        return x + self.mlp(self.norm2(x))


class JEPAPredictor(nn.Module):
    """I-JEPA transformer predictor: D_pred bottleneck with mask tokens.

    context_emb (B, L, D_enc) → Proj(D_enc → D_pred)
    → replace target positions with mask_token + pos_embed
    → [TransformerBlock × depth] → LN → Proj(D_pred → D_enc)
    → (B, L, D_enc) predictions at all positions
    """

    def __init__(self, embed_dim: int, predictor_dim: int = 384,
                 depth: int = 4, num_heads: int = 6, max_seq_len: int = 512):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(embed_dim, predictor_dim),
            nn.LayerNorm(predictor_dim),
        )
        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, predictor_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.blocks = nn.ModuleList([
            _PredictorBlock(predictor_dim, num_heads) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(predictor_dim)
        self.output_proj = nn.Linear(predictor_dim, embed_dim)

    def forward(self, context_token_emb: torch.Tensor, target_mask: torch.Tensor) -> torch.Tensor:
        B, L, _ = context_token_emb.shape
        x = self.input_proj(context_token_emb)                          # (B, L, D_pred)
        target_float = target_mask.unsqueeze(-1).float()                # (B, L, 1)
        x = x * (1.0 - target_float) + self.mask_token.expand(B, L, -1) * target_float
        x = x + self.pos_embed[:, :L, :]
        for block in self.blocks:
            x = block(x)
        return self.output_proj(self.norm(x))                           # (B, L, D_enc)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Multi-Block Masking (I-JEPA style) + Curriculum
# ═══════════════════════════════════════════════════════════════════════════════

def multi_block_mask_1d(
    seq_len: int, mask_ratio: float, num_target_blocks: int,
    min_block_len: int, eligible: torch.Tensor, device: torch.device,
) -> torch.Tensor:
    """Generate I-JEPA multi-block target masks (non-overlapping contiguous spans)."""
    B = eligible.shape[0]
    target_mask = torch.zeros(B, seq_len, dtype=torch.bool, device=device)
    elig_lens = eligible.sum(dim=1)

    for b in range(B):
        L_elig = elig_lens[b].item()
        n_target = int(L_elig * mask_ratio)
        if L_elig < min_block_len * 2 or n_target < min_block_len:
            blen = max(1, min(min_block_len, int(L_elig) - 1))
            if blen > 0 and int(L_elig) > blen:
                start = torch.randint(0, int(L_elig) - blen + 1, (1,)).item()
                target_mask[b, start:start + blen] = True
            continue

        per_block = max(min_block_len, n_target // num_target_blocks)
        remaining = n_target
        occupied = torch.zeros(int(L_elig), dtype=torch.bool)

        for _ in range(num_target_blocks):
            if remaining < min_block_len:
                break
            blen = min(max(min_block_len, per_block), remaining, int(L_elig))
            placed = False
            for _attempt in range(50):
                max_start = int(L_elig) - blen
                if max_start < 0:
                    break
                start = torch.randint(0, max_start + 1, (1,)).item()
                if not occupied[start:start + blen].any():
                    occupied[start:start + blen] = True
                    target_mask[b, start:start + blen] = True
                    remaining -= blen
                    placed = True
                    break
            if not placed:
                free = (~occupied).nonzero(as_tuple=True)[0]
                if len(free) >= min_block_len:
                    start = free[0].item()
                    actual = min(blen, len(free), int(L_elig) - start)
                    actual = max(1, actual)
                    occupied[start:start + actual] = True
                    target_mask[b, start:start + actual] = True
                    remaining -= actual

    target_mask &= eligible
    return target_mask


def curriculum_masking_params(epoch: int, total_epochs: int, cfg: MaskingConfig) -> Tuple[float, int]:
    """Cosine ramp from start → end values."""
    progress = epoch / max(total_epochs - 1, 1)
    t = 0.5 * (1.0 - math.cos(math.pi * progress))
    mask_ratio = cfg.mask_ratio_start + t * (cfg.mask_ratio_end - cfg.mask_ratio_start)
    min_block_len = int(cfg.min_block_len_start + t * (cfg.min_block_len_end - cfg.min_block_len_start))
    mask_ratio = max(cfg.min_mask_ratio, min(cfg.max_mask_ratio, mask_ratio))
    return mask_ratio, max(1, min_block_len)


# ═══════════════════════════════════════════════════════════════════════════════
# 6. GC Content + Adversary + RC Consistency
# ═══════════════════════════════════════════════════════════════════════════════

def compute_gc_content(tokens: torch.Tensor, pad_token_id: int = 0,
                       gc_token_ids: Optional[set] = None) -> torch.Tensor:
    """Compute per-sequence GC fraction. Returns (B,) tensor."""
    non_pad = (tokens != pad_token_id)
    lengths = non_pad.sum(dim=1).float().clamp(min=1)
    gc_mask = torch.zeros_like(tokens, dtype=torch.bool)
    if gc_token_ids:
        for tid in gc_token_ids:
            gc_mask |= (tokens == tid)
    return (gc_mask & non_pad).sum(dim=1).float() / lengths


class _GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


class GCAdversary(nn.Module):
    """Gradient-reversal adversary for GC content debiasing."""
    def __init__(self, embed_dim: int = 576, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, embeddings: torch.Tensor, lambda_: float = 1.0) -> torch.Tensor:
        x = _GradReverse.apply(embeddings, lambda_)
        return self.net(x).squeeze(-1)

    @staticmethod
    def ganin_lambda(epoch: int, total_epochs: int, gamma: float = 10.0) -> float:
        p = epoch / max(total_epochs - 1, 1)
        return 2.0 / (1.0 + math.exp(-gamma * p)) - 1.0


def reverse_complement_tokens(tokens: torch.Tensor, comp_map: dict, pad_token_id: int = 0) -> torch.Tensor:
    """Reverse complement a batch of tokenized sequences."""
    rc = tokens.clone()
    for orig, comp in comp_map.items():
        rc[tokens == orig] = comp
    # Reverse non-pad portion
    for b in range(rc.shape[0]):
        L = (rc[b] != pad_token_id).sum().item()
        if L > 0:
            rc[b, :L] = rc[b, :L].flip(0)
    return rc


def gc_correlation(tokens: torch.Tensor, embeddings: torch.Tensor,
                   pad_token_id: int = 0, gc_token_ids: Optional[set] = None) -> Tuple[float, float]:
    """Pearson |r| between GC content and first PC of embeddings."""
    gc = compute_gc_content(tokens, pad_token_id, gc_token_ids).numpy()
    emb_np = embeddings.float().numpy()
    try:
        from numpy.linalg import svd
        _, _, Vt = svd(emb_np - emb_np.mean(axis=0), full_matrices=False)
        pc1 = emb_np @ Vt[0]
        r = float(np.corrcoef(gc, pc1)[0, 1])
        return abs(r), r
    except Exception:
        return 0.0, 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Cas12aJEPA Model — I-JEPA for DNA
# ═══════════════════════════════════════════════════════════════════════════════

def compute_prediction_loss(pred: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, dict]:
    """MSE + cosine similarity monitoring at target positions."""
    loss = F.mse_loss(pred, target)
    with torch.no_grad():
        cos_sim = F.cosine_similarity(pred, target, dim=-1).mean().item()
    return loss, {"pred_mse": loss.item(), "cos_sim": cos_sim}


class Cas12aJEPA(nn.Module):
    """I-JEPA for DNA sequences with per-token prediction.

    1. Multi-block masking → target blocks
    2. Context encoder: masked input → (B, L, D)
    3. Predictor: context + mask tokens → predicted embeddings
    4. Target encoder (EMA): full input → target embeddings
    5. Loss: MSE(pred[mask], target[mask]) + λ·SIGReg(ctx_pooled)
    """

    def __init__(self, encoder: nn.Module, config: JEPAConfig):
        super().__init__()
        self.config = config
        self.context_encoder = encoder
        self.target_encoder = copy.deepcopy(encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False

        embed_dim = encoder.embed_dim
        self.predictor = JEPAPredictor(
            embed_dim=embed_dim, predictor_dim=config.predictor_dim,
            depth=config.predictor_depth, num_heads=config.predictor_num_heads,
            max_seq_len=config.max_seq_len,
        )
        self.sigreg = SIGRegVectorized(
            num_slices=config.sigreg_num_slices,
            num_points=config.sigreg_num_points,
        )
        self._ema_decay = config.ema_decay_start

    def set_ema_decay(self, progress: float) -> float:
        t0, t1 = self.config.ema_decay_start, self.config.ema_decay_end
        self._ema_decay = t1 - (t1 - t0) * (1 + math.cos(math.pi * progress)) / 2
        return self._ema_decay

    @torch.no_grad()
    def update_ema(self) -> None:
        tau = self._ema_decay
        for tp, cp in zip(self.target_encoder.parameters(), self.context_encoder.parameters()):
            tp.data.mul_(tau).add_(cp.data, alpha=1.0 - tau)

    def forward(self, tokens: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
                mask_ratio: float = 0.30, min_block_len: int = 5,
                pad_token_id: int = 0) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        B, L = tokens.shape
        device = tokens.device
        mcfg = self.config.masking
        effective_ratio = max(mcfg.min_mask_ratio, min(mask_ratio, mcfg.max_mask_ratio))
        eligible = tokens != pad_token_id

        # Multi-block masking
        target_mask = multi_block_mask_1d(
            L, effective_ratio, mcfg.num_target_blocks,
            min_block_len, eligible, device,
        )

        # Context encoder: masked input
        masked_tokens = tokens.clone()
        masked_tokens[target_mask] = pad_token_id
        _, ctx_emb = self.context_encoder(masked_tokens, attention_mask)  # (B, L, D)

        # Predictor: context + mask tokens → predictions at all positions
        pred_all = self.predictor(ctx_emb, target_mask)  # (B, L, D)

        # Target encoder: full input
        with torch.no_grad():
            _, tgt_emb = self.target_encoder(tokens, attention_mask)  # (B, L, D)

        # Extract target positions only
        pred_emb = pred_all[target_mask]      # (N_masked, D)
        target_emb = tgt_emb[target_mask]     # (N_masked, D)

        # Context-pooled (for SIGReg + auxiliary losses)
        visible = eligible & ~target_mask
        vis_float = visible.unsqueeze(-1).float()
        context_pooled = (ctx_emb * vis_float).sum(dim=1) / vis_float.sum(dim=1).clamp(min=1)

        # Target-pooled (for sequence-level loss)
        elig_float = eligible.unsqueeze(-1).float()
        target_pooled = (tgt_emb * elig_float).sum(dim=1) / elig_float.sum(dim=1).clamp(min=1)

        info = {
            "mask": target_mask, "n_masked": target_mask.sum().item(),
            "n_eligible": eligible.sum().item(), "ema_decay": self._ema_decay,
            "context_pooled": context_pooled,
            "target_pooled": target_pooled.detach(),
            "effective_mask_ratio": effective_ratio,
        }
        return pred_emb, target_emb, info

    @torch.no_grad()
    def encode(self, tokens, attention_mask=None, use_target=True):
        encoder = self.target_encoder if use_target else self.context_encoder
        pooled, _ = encoder(tokens, attention_mask=attention_mask)
        return pooled


# ═══════════════════════════════════════════════════════════════════════════════
# 8. RankMe + UMAP (GC-colored) + t-SNE (fixed)
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def compute_rankme(embeddings: torch.Tensor) -> float:
    """RankMe: effective dimensionality via entropy of singular values."""
    X = embeddings.float()
    X = X - X.mean(dim=0, keepdim=True)
    try:
        s = torch.linalg.svdvals(X)
    except Exception:
        _, s, _ = torch.svd(X)
    s = s[s > 1e-7]
    p = s / s.sum()
    entropy = -(p * p.log()).sum().item()
    return math.exp(entropy)


@torch.no_grad()
def generate_viz(embeddings_np: np.ndarray, gc_contents: np.ndarray,
                 save_path: str, epoch: int):
    """Generate UMAP (GC-colored) + t-SNE plots."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    # UMAP colored by GC content
    try:
        import umap
        reducer = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.3,
                            metric='cosine', random_state=42)
        umap_emb = reducer.fit_transform(embeddings_np)
        sc = axes[0].scatter(umap_emb[:, 0], umap_emb[:, 1],
                             c=gc_contents, cmap='RdYlBu_r', s=3, alpha=0.6)
        plt.colorbar(sc, ax=axes[0], label='GC content')
        axes[0].set_title(f'UMAP — Epoch {epoch} (colored by GC%)')
    except Exception as e:
        axes[0].text(0.5, 0.5, f'UMAP failed: {e}', transform=axes[0].transAxes,
                     ha='center', fontsize=10)
        axes[0].set_title(f'UMAP — Epoch {epoch}')

    # t-SNE (fixed: max_iter, not n_iter)
    try:
        from sklearn.manifold import TSNE
        tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42)
        tsne_emb = tsne.fit_transform(embeddings_np)
        sc2 = axes[1].scatter(tsne_emb[:, 0], tsne_emb[:, 1],
                              c=gc_contents, cmap='RdYlBu_r', s=3, alpha=0.6)
        plt.colorbar(sc2, ax=axes[1], label='GC content')
        axes[1].set_title(f't-SNE — Epoch {epoch} (colored by GC%)')
    except Exception as e:
        axes[1].text(0.5, 0.5, f't-SNE failed: {e}', transform=axes[1].transAxes,
                     ha='center', fontsize=10)
        axes[1].set_title(f't-SNE — Epoch {epoch}')

    for ax in axes:
        ax.set_xlabel('Dim 1')
        ax.set_ylabel('Dim 2')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved viz: {save_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# 9. Dataset wrapper + BPE tokenizer loading
# ═══════════════════════════════════════════════════════════════════════════════

class BPEPretrainDataset(Dataset):
    """Load pretrain_2M.csv with BPE tokenizer. Returns (token_ids, genome_id)."""

    def __init__(self, csv_path: str, tokenizer_path: str, max_len: int = 512):
        import pandas as pd
        self.df = pd.read_csv(csv_path)
        self.max_len = max_len

        # Load BPE tokenizer
        import json
        with open(tokenizer_path) as f:
            tok_data = json.load(f)

        self.vocab = tok_data.get('model', tok_data).get('vocab', tok_data.get('vocab', {}))
        self.merges = tok_data.get('model', tok_data).get('merges', tok_data.get('merges', []))
        self.pad_id = 0

        # Try tokenizers library first, fall back to simple
        try:
            from tokenizers import Tokenizer
            self.tokenizer = Tokenizer.from_file(tokenizer_path)
            self._encode = self._encode_tokenizers
        except ImportError:
            self._encode = self._encode_simple

        # Genome IDs
        if 'genome' in self.df.columns:
            genomes = self.df['genome'].values
            unique = sorted(set(genomes))
            self.genome_map = {g: i for i, g in enumerate(unique)}
            self.genome_ids = [self.genome_map[g] for g in genomes]
            self.n_genomes = len(unique)
        else:
            self.genome_ids = [0] * len(self.df)
            self.n_genomes = 0

        # Identify GC token IDs
        self.gc_token_ids = set()
        for key, val in self.vocab.items():
            key_upper = key.upper().replace('Ġ', '').replace('▁', '')
            if key_upper and all(c in 'GC' for c in key_upper):
                self.gc_token_ids.add(val)
        if not self.gc_token_ids:
            self.gc_token_ids = {2, 3}

        seq_col = 'sequence' if 'sequence' in self.df.columns else self.df.columns[0]
        self.sequences = self.df[seq_col].values
        print(f"  Dataset: {len(self)} sequences, {self.n_genomes} genomes")
        print(f"  GC token IDs: {len(self.gc_token_ids)} tokens identified")

    def _encode_tokenizers(self, seq: str) -> List[int]:
        enc = self.tokenizer.encode(seq)
        return enc.ids[:self.max_len]

    def _encode_simple(self, seq: str) -> List[int]:
        # Fallback: character-level with vocab lookup
        ids = []
        for c in seq.upper():
            ids.append(self.vocab.get(c, 1))  # 1 = UNK
        return ids[:self.max_len]

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        ids = self._encode(str(self.sequences[idx]))
        # Pad to max_len
        if len(ids) < self.max_len:
            ids = ids + [self.pad_id] * (self.max_len - len(ids))
        return torch.tensor(ids, dtype=torch.long), self.genome_ids[idx]


# ═══════════════════════════════════════════════════════════════════════════════
# 10. Training Utilities
# ═══════════════════════════════════════════════════════════════════════════════

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cosine_lr(optimizer, step, total_steps, warmup_steps, peak_lr, min_lr=1e-6):
    if step < warmup_steps:
        lr = peak_lr * (step / max(warmup_steps, 1))
    else:
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        lr = min_lr + 0.5 * (peak_lr - min_lr) * (1 + math.cos(math.pi * progress))
    for pg in optimizer.param_groups:
        pg["lr"] = lr
    return lr


def fmt_time(s):
    h, m, sec = int(s // 3600), int((s % 3600) // 60), int(s % 60)
    return f"{h}:{m:02d}:{sec:02d}"


def build_complement_map_bpe(dataset):
    """Build complement map for BPE tokens."""
    comp = {}
    vocab = dataset.vocab
    base_pairs = [('A', 'T'), ('C', 'G')]
    for b1, b2 in base_pairs:
        id1 = vocab.get(b1) or vocab.get(b1.lower())
        id2 = vocab.get(b2) or vocab.get(b2.lower())
        if id1 is not None and id2 is not None:
            comp[id1] = id2
            comp[id2] = id1
    return comp


# ═══════════════════════════════════════════════════════════════════════════════
# 11. Evaluation
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_epoch(model, dataloader, device, gc_token_ids, max_batches=30):
    model.eval()
    all_emb, all_tok = [], []
    for i, batch in enumerate(dataloader):
        if i >= max_batches:
            break
        tokens = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)
        attn = model.context_encoder.get_attention_mask(tokens)
        emb = model.encode(tokens, attention_mask=attn)
        all_emb.append(emb.cpu())
        all_tok.append(tokens.cpu())

    embeddings = torch.cat(all_emb, dim=0)
    tokens = torch.cat(all_tok, dim=0)

    rankme = compute_rankme(embeddings)
    gc_abs, gc_raw = gc_correlation(tokens, embeddings, pad_token_id=0, gc_token_ids=gc_token_ids)
    pred_std = embeddings.std(dim=0).mean().item()
    embed_norm = embeddings.norm(dim=1).mean().item()
    sigreg_score = model.sigreg.gaussianity_score(embeddings.float().to(device))

    return {
        "rankme": rankme, "gc_abs_r": gc_abs, "gc_raw_r": gc_raw,
        "pred_std": pred_std, "norm": embed_norm, "sigreg_eval": sigreg_score,
        "embeddings": embeddings, "tokens": tokens,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 12. Main Training Loop
# ═══════════════════════════════════════════════════════════════════════════════

def pretrain(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'═' * 70}")
    print(f"  B-JEPA v4.5 — I-JEPA + SIGReg for Bacterial DNA")
    print(f"{'═' * 70}")
    print(f"  Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    # ── Dataset ──
    dataset = BPEPretrainDataset(args.data_path, args.tokenizer_path, args.max_seq_len)
    pad_id = dataset.pad_id
    gc_token_ids = dataset.gc_token_ids
    vocab_size = len(dataset.vocab)
    comp_map = build_complement_map_bpe(dataset)

    dl = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                    num_workers=args.num_workers, pin_memory=True, drop_last=True,
                    prefetch_factor=2)

    # ── Model ──
    cfg = JEPAConfig(
        predictor_dim=args.predictor_dim,
        predictor_depth=args.predictor_depth,
        predictor_num_heads=args.predictor_num_heads,
        max_seq_len=args.max_seq_len,
        ema_decay_start=args.ema_decay_start,
        ema_decay_end=args.ema_decay_end,
        masking=MaskingConfig(
            mask_ratio_start=args.mask_ratio_start,
            mask_ratio_end=args.mask_ratio_end,
            num_target_blocks=args.num_target_blocks,
            min_block_len_start=args.min_block_len_start,
            min_block_len_end=args.min_block_len_end,
        ),
        sigreg_num_slices=args.sigreg_num_slices,
    )

    encoder = DNATransformerEncoder(
        vocab_size=vocab_size, embed_dim=args.embed_dim,
        num_layers=args.num_layers, num_heads=args.num_heads,
        ff_dim=args.ff_dim, max_seq_len=args.max_seq_len,
        dropout=0.1, pad_token_id=pad_id,
    )
    model = Cas12aJEPA(encoder, config=cfg).to(device)
    gc_adv = GCAdversary(embed_dim=args.embed_dim, hidden_dim=64).to(device)

    n_enc = sum(p.numel() for p in model.context_encoder.parameters())
    n_pred = sum(p.numel() for p in model.predictor.parameters())
    n_total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  Encoder: {n_enc/1e6:.2f}M  Predictor: {n_pred/1e6:.2f}M  Total: {n_total/1e6:.2f}M")
    print(f"  Architecture: {args.num_layers}L × {args.embed_dim}D × {args.num_heads}H")
    print(f"  Predictor: {args.predictor_depth}L × {args.predictor_dim}D × {args.predictor_num_heads}H")

    # ── Optimizer ──
    opt = torch.optim.AdamW([
        {"params": model.context_encoder.parameters()},
        {"params": model.predictor.parameters()},
        {"params": gc_adv.parameters(), "lr": args.lr * 5},
    ], lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95))

    steps_per_epoch = len(dl)
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = args.warmup_epochs * steps_per_epoch
    grad_accum = args.grad_accum_steps

    # AMP
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler('cuda', enabled=(use_amp and amp_dtype == torch.float16))

    # Output dirs
    ckpt_dir = Path(args.output_dir) / "checkpoints" / "v4.5"
    viz_dir = ckpt_dir / "viz"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  Training: {args.epochs} epochs, warmup={args.warmup_epochs}")
    print(f"  LR: {args.lr:.1e} → {args.min_lr:.1e} (cosine)")
    print(f"  Masking: {args.mask_ratio_start:.0%} → {args.mask_ratio_end:.0%} (curriculum)")
    print(f"  SIGReg λ={args.sigreg_weight}  seq_pred={args.seq_pred_weight}")
    print(f"  RC={args.rc_weight}  GC_adv={args.gc_adv_weight}")
    print(f"  Steps/epoch: {steps_per_epoch}  Total: {total_steps}")
    print(f"{'═' * 70}\n")

    # W&B
    use_wandb = HAS_WANDB and not args.no_wandb
    if use_wandb:
        wandb.init(project=args.wandb_project,
                   name=args.wandb_run_name or f"bjepa-v4.5-{args.num_layers}L{args.embed_dim}D",
                   config=vars(args))

    global_step = 0
    best_loss = float('inf')
    training_start = time.time()

    for epoch in range(args.epochs):
        epoch_start = time.time()
        model.train()
        gc_adv.train()

        progress = epoch / max(args.epochs - 1, 1)
        current_ema = model.set_ema_decay(progress)
        mr, mbl = curriculum_masking_params(epoch, args.epochs, cfg.masking)
        lam_gc = GCAdversary.ganin_lambda(epoch, args.epochs) if args.gc_adv_weight > 0 else 0.0

        epoch_losses = []
        epoch_comp = {"pred_mse": [], "sigreg": [], "seq_pred": [], "rc": [], "gc_adv": [], "cos_sim": []}
        opt.zero_grad(set_to_none=True)

        pbar = tqdm(dl, desc=f"Epoch {epoch+1}/{args.epochs}", leave=True, ncols=140)

        for step, batch in enumerate(pbar, start=1):
            global_step += 1
            lr = cosine_lr(opt, global_step, total_steps, warmup_steps, args.lr, args.min_lr)

            tokens, gids = batch[0].to(device), batch[1].to(device)
            attn = model.context_encoder.get_attention_mask(tokens)

            ctx = torch.autocast("cuda", dtype=amp_dtype) if use_amp else nullcontext()
            with ctx:
                # Forward
                pred, tgt, info = model(tokens, attention_mask=attn,
                                        mask_ratio=mr, min_block_len=mbl, pad_token_id=pad_id)

                # Loss 1: Per-token prediction MSE
                pred_loss, pred_met = compute_prediction_loss(pred, tgt)
                total = pred_loss

                # Loss 2: SIGReg
                sig_val = 0.0
                if args.sigreg_weight > 0:
                    sig_loss = model.sigreg(info["context_pooled"])
                    total = total + args.sigreg_weight * sig_loss
                    sig_val = sig_loss.item()

                # Loss 3: Sequence-level prediction
                sp_val = 0.0
                if args.seq_pred_weight > 0:
                    sp_loss = F.mse_loss(info["context_pooled"], info["target_pooled"])
                    total = total + args.seq_pred_weight * sp_loss
                    sp_val = sp_loss.item()

                # Loss 4: RC consistency
                rc_val = 0.0
                if args.rc_weight > 0 and len(comp_map) >= 2:
                    rc_tok = reverse_complement_tokens(tokens, comp_map, pad_id)
                    rc_pooled, _ = model.context_encoder(rc_tok, attention_mask=attn)
                    rc_loss = F.mse_loss(info["context_pooled"], rc_pooled)
                    total = total + args.rc_weight * rc_loss
                    rc_val = rc_loss.item()

                # Loss 5: GC adversary
                ga_val = 0.0
                if args.gc_adv_weight > 0:
                    gc_t = compute_gc_content(tokens, pad_id, gc_token_ids)
                    gc_p = gc_adv(info["context_pooled"], lambda_=lam_gc)
                    ga_loss = F.mse_loss(gc_p, gc_t)
                    total = total + args.gc_adv_weight * ga_loss
                    ga_val = ga_loss.item()

                scaled = total / grad_accum

            if scaler.is_enabled():
                scaler.scale(scaled).backward()
            else:
                scaled.backward()

            should_step = (step % grad_accum == 0) or (step == steps_per_epoch)
            if should_step:
                if scaler.is_enabled():
                    scaler.unscale_(opt)
                for params in (model.context_encoder.parameters(), model.predictor.parameters(), gc_adv.parameters()):
                    torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
                if scaler.is_enabled():
                    scaler.step(opt)
                    scaler.update()
                else:
                    opt.step()
                opt.zero_grad(set_to_none=True)
                model.update_ema()

            epoch_losses.append(total.item())
            epoch_comp["pred_mse"].append(pred_met["pred_mse"])
            epoch_comp["cos_sim"].append(pred_met["cos_sim"])
            epoch_comp["sigreg"].append(sig_val)
            epoch_comp["seq_pred"].append(sp_val)
            epoch_comp["rc"].append(rc_val)
            epoch_comp["gc_adv"].append(ga_val)

            if step % args.log_every == 0:
                pbar.set_postfix({
                    "loss": f"{total.item():.4f}",
                    "mse": f"{pred_met['pred_mse']:.4f}",
                    "cos": f"{pred_met['cos_sim']:.3f}",
                    "sig": f"{sig_val:.4f}",
                    "mr": f"{mr:.2f}",
                    "lr": f"{lr:.1e}",
                })

                if use_wandb:
                    wandb.log({
                        "train/total_loss": total.item(),
                        "train/pred_mse": pred_met["pred_mse"],
                        "train/cos_sim": pred_met["cos_sim"],
                        "train/sigreg": sig_val,
                        "train/seq_pred": sp_val,
                        "train/rc": rc_val,
                        "train/gc_adv": ga_val,
                        "schedule/lr": lr,
                        "schedule/mask_ratio": mr,
                    }, step=global_step)

        # ── Epoch Eval ──
        epoch_time = time.time() - epoch_start
        avg = {k: sum(v) / max(len(v), 1) for k, v in epoch_comp.items()}
        avg_loss = sum(epoch_losses) / max(len(epoch_losses), 1)

        eval_met = evaluate_epoch(model, dl, device, gc_token_ids, max_batches=30)

        elapsed = time.time() - training_start
        eta = (elapsed / max(epoch + 1, 1)) * (args.epochs - epoch - 1)

        print(f"\n  Epoch {epoch+1}/{args.epochs} ({epoch_time:.1f}s, ETA {fmt_time(eta)})")
        print(f"  Total: {avg_loss:.4f}  pred_mse={avg['pred_mse']:.4f}  cos_sim={avg['cos_sim']:.3f}")
        print(f"  SIGReg: {avg['sigreg']:.5f}  seq_pred={avg['seq_pred']:.4f}  "
              f"RC={avg['rc']:.4f}  GC_adv={avg['gc_adv']:.5f}")
        print(f"  RankMe: {eval_met['rankme']:.1f}/{args.embed_dim}  "
              f"GC |r|={eval_met['gc_abs_r']:.3f}  std={eval_met['pred_std']:.3f}  "
              f"norm={eval_met['norm']:.1f}  SIGReg_eval={eval_met['sigreg_eval']:.4f}")
        print(f"  Mask: {mr:.3f}  LR={lr:.2e}  EMA τ={current_ema:.4f}")

        # RankMe collapse warning
        if eval_met['rankme'] < 50:
            print(f"  ⚠ WARNING: RankMe={eval_met['rankme']:.1f} < 50 — possible collapse!")
        elif eval_met['rankme'] < 200:
            print(f"  ⚡ WATCH: RankMe={eval_met['rankme']:.1f} — monitoring")
        else:
            print(f"  ✅ HEALTHY: RankMe={eval_met['rankme']:.1f}")

        # Generate UMAP + t-SNE colored by GC
        try:
            emb_np = eval_met['embeddings'].float().numpy()
            gc_np = compute_gc_content(eval_met['tokens'], pad_id, gc_token_ids).numpy()
            # Subsample for viz
            n_viz = min(3000, len(emb_np))
            idx = np.random.choice(len(emb_np), n_viz, replace=False)
            viz_path = str(viz_dir / f"embeddings_epoch{epoch+1:03d}.png")
            generate_viz(emb_np[idx], gc_np[idx], viz_path, epoch + 1)
        except Exception as e:
            print(f"  Viz failed: {e}")

        if use_wandb:
            log_dict = {
                "epoch/total_loss": avg_loss,
                "epoch/pred_mse": avg["pred_mse"],
                "epoch/cos_sim": avg["cos_sim"],
                "epoch/sigreg": avg["sigreg"],
                "health/rankme": eval_met["rankme"],
                "health/rankme_ratio": eval_met["rankme"] / args.embed_dim,
                "health/pred_std": eval_met["pred_std"],
                "health/gc_abs_r": eval_met["gc_abs_r"],
                "health/sigreg_eval": eval_met["sigreg_eval"],
                "schedule/ema_tau": current_ema,
                "schedule/mask_ratio": mr,
                "epoch": epoch + 1,
            }
            if os.path.exists(str(viz_dir / f"embeddings_epoch{epoch+1:03d}.png")):
                log_dict["viz/embeddings"] = wandb.Image(
                    str(viz_dir / f"embeddings_epoch{epoch+1:03d}.png"))
            wandb.log(log_dict, step=global_step)

        if avg_loss < best_loss:
            best_loss = avg_loss

        # ── Checkpoint ──
        if (epoch + 1) % args.save_every == 0 or (epoch + 1) == args.epochs:
            ckpt_path = ckpt_dir / f"epoch{epoch+1:04d}.pt"
            torch.save({
                "epoch": epoch, "global_step": global_step,
                "encoder_state_dict": model.context_encoder.state_dict(),
                "target_encoder_state_dict": model.target_encoder.state_dict(),
                "predictor_state_dict": model.predictor.state_dict(),
                "gc_adversary_state_dict": gc_adv.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "scaler_state_dict": scaler.state_dict() if scaler.is_enabled() else None,
                "loss": avg_loss,
                "metrics": eval_met | {"avg_pred_mse": avg["pred_mse"], "avg_cos_sim": avg["cos_sim"]},
                "config": vars(args),
            }, ckpt_path)
            print(f"  Saved: {ckpt_path}")

    total_time = time.time() - training_start
    print(f"\n{'═' * 70}")
    print(f"  B-JEPA v4.5 complete in {fmt_time(total_time)}")
    print(f"  Best loss: {best_loss:.4f}  Final RankMe: {eval_met['rankme']:.1f}")
    print(f"{'═' * 70}")
    if use_wandb:
        wandb.finish()


# ═══════════════════════════════════════════════════════════════════════════════
# 13. CLI
# ═══════════════════════════════════════════════════════════════════════════════

def build_parser():
    p = argparse.ArgumentParser(description="B-JEPA v4.5 Pretraining")
    g = p.add_argument_group("Data")
    g.add_argument("--data-path", default="data/processed/pretrain_2M.csv")
    g.add_argument("--tokenizer-path", default="data/tokenizer/bpe_4096.json")
    g.add_argument("--output-dir", default="outputs")
    g.add_argument("--seed", type=int, default=42)

    g = p.add_argument_group("Training")
    g.add_argument("--epochs", type=int, default=30)
    g.add_argument("--batch-size", type=int, default=64)
    g.add_argument("--lr", type=float, default=3e-4)
    g.add_argument("--min-lr", type=float, default=1e-6)
    g.add_argument("--warmup-epochs", type=int, default=5)
    g.add_argument("--weight-decay", type=float, default=0.05)
    g.add_argument("--grad-clip", type=float, default=1.0)
    g.add_argument("--grad-accum-steps", type=int, default=2)

    g = p.add_argument_group("Encoder")
    g.add_argument("--embed-dim", type=int, default=576)
    g.add_argument("--num-layers", type=int, default=12)
    g.add_argument("--num-heads", type=int, default=9)
    g.add_argument("--ff-dim", type=int, default=2304)
    g.add_argument("--max-seq-len", type=int, default=512)

    g = p.add_argument_group("Predictor")
    g.add_argument("--predictor-dim", type=int, default=384)
    g.add_argument("--predictor-depth", type=int, default=4)
    g.add_argument("--predictor-num-heads", type=int, default=6)
    g.add_argument("--ema-decay-start", type=float, default=0.996)
    g.add_argument("--ema-decay-end", type=float, default=1.0)

    g = p.add_argument_group("Masking")
    g.add_argument("--num-target-blocks", type=int, default=4)
    g.add_argument("--mask-ratio-start", type=float, default=0.15)
    g.add_argument("--mask-ratio-end", type=float, default=0.50)
    g.add_argument("--min-block-len-start", type=int, default=3)
    g.add_argument("--min-block-len-end", type=int, default=15)

    g = p.add_argument_group("Losses")
    g.add_argument("--sigreg-weight", type=float, default=1.0)
    g.add_argument("--sigreg-num-slices", type=int, default=512)
    g.add_argument("--seq-pred-weight", type=float, default=0.5)
    g.add_argument("--rc-weight", type=float, default=0.1)
    g.add_argument("--gc-adv-weight", type=float, default=1.0)

    g = p.add_argument_group("Logging")
    g.add_argument("--save-every", type=int, default=5)
    g.add_argument("--log-every", type=int, default=50)
    g.add_argument("--num-workers", type=int, default=4)
    g.add_argument("--no-wandb", action="store_true")
    g.add_argument("--wandb-project", default="bdna-jepa")
    g.add_argument("--wandb-run-name", default=None)

    return p


if __name__ == "__main__":
    pretrain(build_parser().parse_args())
