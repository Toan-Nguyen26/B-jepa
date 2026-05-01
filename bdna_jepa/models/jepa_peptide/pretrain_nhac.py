#!/usr/bin/env python3
"""P-JEPA: Peptide JEPA for Non-Histone Lysine Acetylation Site Prediction
===========================================================================

Adapts the B-JEPA architecture (jepa_v6/pretrain_v6.py) to amino acid
sequences for predicting non-histone lysine (K) acetylation sites.

Two-phase training:
  Phase 1 — JEPA pretraining (self-supervised, no labels):
    Train on raw 31-AA sequences using JEPA + MLM objectives.
    Model learns local amino acid context representations.

  Phase 2 — Fine-tuning (supervised):
    Add ClassificationHead on CLS embedding.
    Train with binary cross-entropy on labeled acetylation sites.

Dataset: NHAC (Non-Histone Acetylation Collection) from TransPTM paper.
  - 787 positive (acetylated K), 4707 negative
  - 31 AA windows (+-15 around K), pre-split train/val/test

Baseline: TransPTM (ProtT5 + GNN) — AUC=0.83, AUPRC=0.51
Target:   Beat TransPTM using JEPA pretraining without manual features.

Changes from B-JEPA (pretrain_v6.py):
  - Tokenizer:   BPE 4096 -> char-level 22 (20 AA + PAD + MASK)
  - Positional:  RoPE -> learned embeddings (seq len <= 31, no need for RoPE)
  - Model size:  scaled down for short sequences (embed_dim=128, 4 layers)
  - Masking:     adapted for 31 tokens (2 blocks, min_len=3, ratio=0.30)
  - Removed:     GC adversary (DNA-specific, not needed for peptides)
  - Added:       ClassificationHead for Phase 2 fine-tuning

Usage:
  # Phase 1: pretrain
  python pretrain_nhac.py --phase pretrain --data-path data/nhac/NHAC_deduplicated.csv

  # Phase 2: fine-tune
  python pretrain_nhac.py --phase finetune --checkpoint outputs/nhac/pretrain_best.pt
"""
from __future__ import annotations

import argparse
import copy
import math
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import contextmanager
from torch.utils.data import DataLoader, Dataset


@contextmanager
def contextlib_nullcontext():
    yield


try:
    from sklearn.metrics import roc_auc_score, average_precision_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kw):
        return x

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False



# =============================================================================
# 1. Tokenizer — Character-level amino acid encoding
# =============================================================================

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"  # 20 standard amino acids
PAD_TOKEN_ID  = 0
MASK_TOKEN_ID = 1

# Build vocab: each AA gets a unique integer starting from 2
VOCAB: Dict[str, int] = {aa: i + 2 for i, aa in enumerate(AMINO_ACIDS)}
VOCAB["[PAD]"]  = PAD_TOKEN_ID
VOCAB["[MASK]"] = MASK_TOKEN_ID
VOCAB_SIZE = len(VOCAB)  # 22


def tokenize(sequence: str) -> List[int]:
    """Convert amino acid string to token IDs. Unknown characters -> PAD."""
    return [VOCAB.get(aa, PAD_TOKEN_ID) for aa in sequence.upper()]


# =============================================================================
# 2. Dataset
# =============================================================================

class NHACDataset(Dataset):
    """NHAC non-histone acetylation dataset.

    Each sample is a 31 AA window centered on a lysine (K) residue.
    Label 1 = acetylated, 0 = not acetylated.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        seq_col: str = "seq_31",
        label_col: Optional[str] = "label",
        max_len: int = 31,
    ):
        self.sequences = df[seq_col].tolist()
        self.labels = df[label_col].tolist() if (label_col and label_col in df.columns) else None
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx):
        tokens = tokenize(self.sequences[idx])
        tokens = tokens[:self.max_len]
        tokens += [PAD_TOKEN_ID] * (self.max_len - len(tokens))
        token_tensor = torch.tensor(tokens, dtype=torch.long)

        if self.labels is not None:
            label = torch.tensor(self.labels[idx], dtype=torch.float)
            return token_tensor, label
        return token_tensor


def load_nhac(
    data_path: str,
    seq_col: str = "seq_31",
) -> Tuple[NHACDataset, NHACDataset, NHACDataset]:
    """Load NHAC CSV and return (train, val, test) datasets."""
    df = pd.read_csv(data_path)
    train_df = df[df["set"] == "train"].reset_index(drop=True)
    val_df   = df[df["set"] == "val"].reset_index(drop=True)
    test_df  = df[df["set"] == "test"].reset_index(drop=True)
    return (
        NHACDataset(train_df, seq_col=seq_col),
        NHACDataset(val_df,   seq_col=seq_col),
        NHACDataset(test_df,  seq_col=seq_col),
    )


def compute_pos_weight(train_dataset: NHACDataset) -> torch.Tensor:
    """Compute positive class weight for imbalanced BCE loss (neg/pos ratio)."""
    labels = torch.tensor(train_dataset.labels, dtype=torch.float)
    n_pos = labels.sum().item()
    n_neg = (1 - labels).sum().item()
    return torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float)


# =============================================================================
# 3. Building Blocks
#    Copied from B-JEPA (pretrain_v6.py) — RoPE removed, rest unchanged.
# =============================================================================

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).type_as(x) * self.weight


class SwiGLU(nn.Module):
    """SwiGLU feedforward (Shazeer, 2020)."""

    def __init__(self, dim: int, ff_dim: int, bias: bool = False):
        super().__init__()
        hidden = int(2 * ff_dim / 3)
        hidden = ((hidden + 7) // 8) * 8
        self.w1 = nn.Linear(dim, hidden, bias=bias)
        self.w2 = nn.Linear(hidden, dim, bias=bias)
        self.w3 = nn.Linear(dim, hidden, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention with QK-Norm. No RoPE (learned pos embeds used instead)."""

    def __init__(self, dim: int, num_heads: int, qk_norm: bool = True, bias: bool = False):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim  = dim // num_heads
        self.qkv      = nn.Linear(dim, 3 * dim, bias=bias)
        self.out_proj = nn.Linear(dim, dim, bias=bias)
        if qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)
        self.qk_norm = qk_norm

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, L, D = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        attn_mask = None
        if key_padding_mask is not None:
            attn_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            attn_mask = attn_mask.expand(-1, -1, L, -1).to(dtype=q.dtype)
            attn_mask = attn_mask.masked_fill(attn_mask == 1, float("-inf"))
            attn_mask = attn_mask.masked_fill(attn_mask == 0, 0.0)

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        return self.out_proj(out.transpose(1, 2).reshape(B, L, D))


class CrossAttention(nn.Module):
    """Cross-attention: queries attend to key-value context."""

    def __init__(self, dim: int, num_heads: int, qk_norm: bool = True, bias: bool = False):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim  = dim // num_heads
        self.q_proj   = nn.Linear(dim, dim, bias=bias)
        self.kv_proj  = nn.Linear(dim, 2 * dim, bias=bias)
        self.out_proj = nn.Linear(dim, dim, bias=bias)
        if qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)
        self.qk_norm = qk_norm

    def forward(self, queries: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        B, Lq, D = queries.shape
        Lk = context.shape[1]

        q  = self.q_proj(queries).reshape(B, Lq, self.num_heads, self.head_dim).transpose(1, 2)
        kv = self.kv_proj(context).reshape(B, Lk, 2, self.num_heads, self.head_dim)
        k, v = kv.unbind(2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        out = F.scaled_dot_product_attention(q, k, v)
        return self.out_proj(out.transpose(1, 2).reshape(B, Lq, D))


class TransformerBlock(nn.Module):
    """Pre-norm transformer block: RMSNorm + Attention + RMSNorm + SwiGLU."""

    def __init__(self, dim: int, num_heads: int, ff_dim: int,
                 qk_norm: bool = True, bias: bool = False):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn  = MultiHeadAttention(dim, num_heads, qk_norm=qk_norm, bias=bias)
        self.norm2 = RMSNorm(dim)
        self.ffn   = SwiGLU(dim, ff_dim, bias=bias)

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None):
        x = x + self.attn(self.norm1(x), key_padding_mask)
        x = x + self.ffn(self.norm2(x))
        return x


# =============================================================================
# 4. Context Encoder — Learned positional embeddings (replaces RoPE)
#    RoPE is designed for long sequences with length generalization.
#    For fixed 31-AA peptide windows, learned embeddings are sufficient
#    and simpler — matching Pham et al. (2023) design choice.
# =============================================================================

class ContextEncoder(nn.Module):
    """Transformer encoder processing ONLY visible (unmasked) tokens.

    Uses learned positional embeddings instead of RoPE.
    Prepends a [CLS] token at position 0 for pooled representation.
    """

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        embed_dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 4,
        ff_dim: int = 256,
        max_seq_len: int = 31,
        dropout: float = 0.1,
        pad_token_id: int = PAD_TOKEN_ID,
        qk_norm: bool = True,
        bias: bool = False,
    ):
        super().__init__()
        self.embed_dim    = embed_dim
        self.pad_token_id = pad_token_id
        self.max_seq_len  = max_seq_len

        self.token_emb   = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_token_id)
        self.pos_embed   = nn.Embedding(max_seq_len + 2, embed_dim)  # +2: CLS pos + buffer
        self.cls_token   = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.embed_drop  = nn.Dropout(dropout)

        self.layers     = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim, qk_norm=qk_norm, bias=bias)
            for _ in range(num_layers)
        ])
        self.final_norm = RMSNorm(embed_dim)
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.token_emb.weight, std=0.02)
        nn.init.normal_(self.pos_embed.weight, std=0.02)
        nn.init.normal_(self.cls_token, std=0.02)
        if self.token_emb.padding_idx is not None:
            self.token_emb.weight.data[self.token_emb.padding_idx].zero_()
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        tokens: torch.Tensor,       # (B, L_vis) visible token IDs
        position_ids: torch.Tensor,  # (B, L_vis) original position indices
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            tokens:       (B, L_vis) — visible token IDs, padded
            position_ids: (B, L_vis) — original positions in full sequence

        Returns:
            cls:    (B, D) — CLS embedding (pooled representation)
            tokens: (B, L_vis, D) — per-token embeddings
        """
        B, L = tokens.shape

        # Token + positional embeddings (position 0 reserved for CLS)
        x = self.token_emb(tokens) + self.pos_embed(position_ids + 1)  # (B, L, D)

        # Prepend CLS at position 0
        cls_emb = self.cls_token.expand(B, -1, -1) + \
                  self.pos_embed(torch.zeros(B, 1, dtype=torch.long, device=tokens.device))
        x = torch.cat([cls_emb, x], dim=1)  # (B, L+1, D)
        x = self.embed_drop(x)

        # Padding mask (True = ignore)
        pad_mask = (tokens == self.pad_token_id)
        cls_mask = torch.zeros(B, 1, device=tokens.device, dtype=torch.bool)
        key_padding_mask = torch.cat([cls_mask, pad_mask], dim=1)

        for layer in self.layers:
            x = layer(x, key_padding_mask)

        x = self.final_norm(x)
        return {
            "cls":    x[:, 0, :],   # (B, D)
            "tokens": x[:, 1:, :],  # (B, L_vis, D)
        }


# =============================================================================
# 5. JEPA Predictor — Cross-attention from target positions to context
#    Identical to B-JEPA predictor but uses learned pos embeddings.
# =============================================================================

class JEPAPredictor(nn.Module):
    """Narrow bottleneck predictor (I-JEPA style).

    Target position queries cross-attend to visible context embeddings,
    then self-attend among themselves. Output projected to encoder dim.
    """

    def __init__(
        self,
        encoder_dim: int = 128,
        predictor_dim: int = 64,
        depth: int = 2,
        num_heads: int = 4,
        max_seq_len: int = 31,
        qk_norm: bool = True,
        bias: bool = False,
    ):
        super().__init__()
        self.context_proj = nn.Sequential(
            nn.Linear(encoder_dim, predictor_dim, bias=bias),
            RMSNorm(predictor_dim),
        )
        self.mask_token       = nn.Parameter(torch.zeros(1, 1, predictor_dim))
        self.pos_embed        = nn.Embedding(max_seq_len + 1, predictor_dim)
        self.context_pos_embed = nn.Embedding(max_seq_len + 1, predictor_dim)
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed.weight, std=0.02)
        nn.init.trunc_normal_(self.context_pos_embed.weight, std=0.02)

        self.blocks = nn.ModuleList()
        for _ in range(depth):
            self.blocks.append(nn.ModuleDict({
                "cross_norm_q":  RMSNorm(predictor_dim),
                "cross_norm_kv": RMSNorm(predictor_dim),
                "cross_attn":    CrossAttention(predictor_dim, num_heads, qk_norm=qk_norm, bias=bias),
                "self_norm":     RMSNorm(predictor_dim),
                "self_attn":     MultiHeadAttention(predictor_dim, num_heads, qk_norm=qk_norm, bias=bias),
                "ffn_norm":      RMSNorm(predictor_dim),
                "ffn":           SwiGLU(predictor_dim, predictor_dim * 2, bias=bias),
            }))

        self.final_norm  = RMSNorm(predictor_dim)
        self.output_proj = nn.Linear(predictor_dim, encoder_dim, bias=bias)

    def forward(
        self,
        context_emb: torch.Tensor,       # (B, L_ctx, D_enc)
        context_positions: torch.Tensor,  # (B, L_ctx)
        target_positions: torch.Tensor,   # (B, L_tgt)
        target_padding_mask: torch.Tensor, # (B, L_tgt) True=pad
    ) -> torch.Tensor:                    # (B, L_tgt, D_enc)
        B, L_tgt = target_positions.shape

        ctx     = self.context_proj(context_emb) + self.context_pos_embed(context_positions)
        queries = self.mask_token.expand(B, L_tgt, -1) + self.pos_embed(target_positions)

        for block in self.blocks:
            queries = queries + block["cross_attn"](
                block["cross_norm_q"](queries),
                block["cross_norm_kv"](ctx),
            )
            queries = queries + block["self_attn"](block["self_norm"](queries))
            queries = queries + block["ffn"](block["ffn_norm"](queries))

        return self.output_proj(self.final_norm(queries))  # (B, L_tgt, D_enc)


# =============================================================================
# 6. MLM Head — Anti-collapse guard
# =============================================================================

class MLMHead(nn.Module):
    def __init__(self, embed_dim: int, vocab_size: int):
        super().__init__()
        self.norm   = RMSNorm(embed_dim)
        self.dense  = nn.Linear(embed_dim, embed_dim)
        self.act    = nn.SiLU()
        self.output = nn.Linear(embed_dim, vocab_size)
        nn.init.normal_(self.dense.weight, std=0.02)
        nn.init.normal_(self.output.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output(self.act(self.dense(self.norm(x))))


# =============================================================================
# 7. SIGReg — Collapse prevention (copied from B-JEPA, scaled down)
# =============================================================================

class SIGReg(nn.Module):
    """Sketched Isotropic Gaussian Regularization + per-dim variance floor."""

    def __init__(self, num_slices: int = 64, num_points: int = 17, var_gamma: float = 1.0):
        super().__init__()
        self.num_slices = num_slices
        self.var_gamma  = var_gamma
        t_max    = 2.0
        t_points = torch.linspace(0, t_max, num_points + 1)[1:]
        self.register_buffer("t_points", t_points)
        dt      = t_max / num_points
        weights = torch.full((num_points,), dt)
        weights[0]  = dt / 2
        weights[-1] = dt / 2
        self.register_buffer("weights", weights)

    def forward(self, embeddings: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        B, D = embeddings.shape
        if B < 4:
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True), {}

        z        = embeddings.float()
        cls_std  = z.std(dim=0)
        var_floor = F.relu(self.var_gamma - cls_std).mean()

        directions = F.normalize(
            torch.randn(D, self.num_slices, device=z.device, dtype=z.dtype), dim=0
        )
        proj = z @ directions
        std  = proj.std(dim=0)
        proj = (proj - proj.mean(dim=0, keepdim=True)) / (std + 1e-8)

        total = torch.tensor(0.0, device=z.device)
        for i, t in enumerate(self.t_points):
            ecf_real = torch.cos(t * proj).mean(dim=0)
            ecf_imag = torch.sin(t * proj).mean(dim=0)
            ecf_sq   = ecf_real ** 2 + ecf_imag ** 2
            tcf      = math.exp(-0.5 * t.item() ** 2)
            total    = total + self.weights[i] * (ecf_sq - 2 * ecf_real * tcf + tcf ** 2).mean()

        loss = total + var_floor
        return loss, {
            "sigreg":    total.item(),
            "var_floor": var_floor.item(),
            "std_mean":  cls_std.mean().item(),
        }


# =============================================================================
# 8. Classification Head — New, for Phase 2 fine-tuning
# =============================================================================

class ClassificationHead(nn.Module):
    """Binary classification head on CLS embedding.

    Matches Pham et al. (2023) prediction head design:
    Linear -> GELU -> Dropout -> Linear -> scalar logit
    """

    def __init__(self, embed_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 1),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, cls: torch.Tensor) -> torch.Tensor:
        """(B, D) -> (B,) raw logits (no sigmoid — use BCEWithLogitsLoss)"""
        return self.net(cls).squeeze(-1)


# =============================================================================
# 9. Multi-block masking — adapted for short 31-token sequences
#    B-JEPA uses 4 blocks, min_len=10, ratio=50-70% for 512-token DNA.
#    For 31-token peptides: 2 blocks, min_len=3, ratio=30%.
# =============================================================================

def multi_block_mask(
    seq_len: int,
    mask_ratio: float,
    num_blocks: int,
    min_block_len: int,
    valid_mask: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Generate multi-block target masks for JEPA pretraining.

    Args:
        seq_len:       Full sequence length (31 for NHAC)
        mask_ratio:    Fraction of valid tokens to mask as targets (0.30)
        num_blocks:    Number of contiguous target blocks (2)
        min_block_len: Minimum block length (3)
        valid_mask:    (B, L) True = non-padding token
        device:        Target device

    Returns:
        target_mask: (B, L) True = target (to be predicted)
    """
    B = valid_mask.shape[0]
    target_mask = torch.zeros(B, seq_len, dtype=torch.bool, device=device)
    valid_lens  = valid_mask.sum(dim=1)

    for b in range(B):
        L_valid  = int(valid_lens[b].item())
        n_target = int(L_valid * mask_ratio)

        if L_valid < min_block_len * 2 or n_target < min_block_len:
            # Sequence too short: mask a single token
            if L_valid > 1:
                start = random.randint(0, L_valid - 1)
                target_mask[b, start] = True
            continue

        per_block = max(min_block_len, n_target // num_blocks)
        remaining = n_target
        occupied  = torch.zeros(L_valid, dtype=torch.bool)

        for _ in range(num_blocks):
            if remaining < min_block_len:
                break
            blen = min(max(min_block_len, per_block), remaining, L_valid)

            placed = False
            for _attempt in range(30):
                start = random.randint(0, max(0, L_valid - blen))
                if not occupied[start:start + blen].any():
                    occupied[start:start + blen] = True
                    target_mask[b, start:start + blen] = True
                    remaining -= blen
                    placed = True
                    break

            if not placed:
                free = (~occupied).nonzero(as_tuple=True)[0]
                if len(free) >= 1:
                    start  = free[0].item()
                    actual = min(blen, int(L_valid) - start)
                    if actual > 0:
                        occupied[start:start + actual] = True
                        target_mask[b, start:start + actual] = True
                        remaining -= actual

    target_mask &= valid_mask
    return target_mask


# =============================================================================
# 10. P-JEPA Model
# =============================================================================

class PeptideJEPA(nn.Module):
    """JEPA model for amino acid sequences.

    Phase 1 (pretraining):  forward_pretrain() — JEPA + MLM, no labels
    Phase 2 (fine-tuning):  forward_finetune() — classification on CLS
    Inference:              encode() — extract CLS embeddings
    """

    def __init__(
        self,
        vocab_size: int      = VOCAB_SIZE,
        embed_dim: int       = 128,
        num_layers: int      = 4,
        num_heads: int       = 4,
        ff_dim: int          = 256,
        max_seq_len: int     = 31,
        predictor_dim: int   = 64,
        predictor_depth: int = 2,
        predictor_heads: int = 4,
        pad_token_id: int    = PAD_TOKEN_ID,
        mask_token_id: int   = MASK_TOKEN_ID,
        ema_start: float     = 0.996,
        ema_end: float       = 1.0,
        mlm_mask_ratio: float = 0.15,
        var_gamma: float      = 1.0,
        # JEPA masking — tuned for 31-token sequences
        jepa_mask_ratio: float  = 0.45,
        jepa_num_blocks: int    = 2,
        jepa_min_block_len: int = 3,
        cls_dropout: float      = 0.1,
    ):
        super().__init__()
        self.pad_token_id      = pad_token_id
        self.mask_token_id     = mask_token_id
        self.embed_dim         = embed_dim
        self.ema_start         = ema_start
        self.ema_end           = ema_end
        self._ema_decay        = ema_start
        self.mlm_mask_ratio    = mlm_mask_ratio
        self.max_seq_len       = max_seq_len
        self.jepa_mask_ratio   = jepa_mask_ratio
        self.jepa_num_blocks   = jepa_num_blocks
        self.jepa_min_block_len = jepa_min_block_len

        enc_kwargs = dict(
            vocab_size=vocab_size, embed_dim=embed_dim,
            num_layers=num_layers, num_heads=num_heads,
            ff_dim=ff_dim, max_seq_len=max_seq_len,
            pad_token_id=pad_token_id,
        )

        # Context encoder (trainable) — sees only visible tokens
        self.context_encoder = ContextEncoder(**enc_kwargs, dropout=0.1)

        # Target encoder (EMA copy, no grad) — sees full sequence
        self.target_encoder  = ContextEncoder(**enc_kwargs, dropout=0.0)
        self.target_encoder.load_state_dict(self.context_encoder.state_dict())
        for p in self.target_encoder.parameters():
            p.requires_grad = False

        self.predictor = JEPAPredictor(
            encoder_dim=embed_dim, predictor_dim=predictor_dim,
            depth=predictor_depth, num_heads=predictor_heads,
            max_seq_len=max_seq_len,
        )
        self.mlm_head  = MLMHead(embed_dim, vocab_size)
        self.sigreg    = SIGReg(num_slices=64, num_points=17, var_gamma=var_gamma)
        self.cls_head  = ClassificationHead(embed_dim, dropout=cls_dropout)

    # ------------------------------------------------------------------
    # EMA
    # ------------------------------------------------------------------

    def set_ema_decay(self, progress: float) -> float:
        t0, t1 = self.ema_start, self.ema_end
        self._ema_decay = t1 - (t1 - t0) * (1 + math.cos(math.pi * progress)) / 2
        return self._ema_decay

    @torch.no_grad()
    def update_ema(self):
        tau = self._ema_decay
        for tp, cp in zip(self.target_encoder.parameters(),
                          self.context_encoder.parameters()):
            tp.data.mul_(tau).add_(cp.data, alpha=1.0 - tau)

    # ------------------------------------------------------------------
    # Internal helpers (identical logic to B-JEPA)
    # ------------------------------------------------------------------

    def _extract_visible_and_target(
        self, tokens: torch.Tensor, target_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B, L  = tokens.shape
        device = tokens.device
        valid        = tokens != self.pad_token_id
        visible_mask = valid & ~target_mask
        pos_ids      = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)

        # Visible tokens
        vis_counts = visible_mask.sum(dim=1)
        max_vis    = max(vis_counts.max().item(), 1)
        vis_order  = (~visible_mask).long().argsort(dim=1, stable=True)
        vis_tokens    = torch.gather(tokens,  1, vis_order)[:, :max_vis].clone()
        vis_positions = torch.gather(pos_ids, 1, vis_order)[:, :max_vis].clone()
        vis_range  = torch.arange(max_vis, device=device).unsqueeze(0)
        vis_pad    = vis_range >= vis_counts.unsqueeze(1)
        vis_tokens[vis_pad]    = self.pad_token_id
        vis_positions[vis_pad] = 0

        # Target positions
        tgt_counts    = target_mask.sum(dim=1)
        max_tgt       = max(tgt_counts.max().item(), 1)
        tgt_order     = (~target_mask).long().argsort(dim=1, stable=True)
        tgt_positions = torch.gather(pos_ids, 1, tgt_order)[:, :max_tgt].clone()
        tgt_range     = torch.arange(max_tgt, device=device).unsqueeze(0)
        tgt_padding   = tgt_range >= tgt_counts.unsqueeze(1)
        tgt_positions[tgt_padding] = 0

        return vis_tokens, vis_positions, tgt_positions, tgt_padding, visible_mask

    def _apply_mlm_mask(
        self, vis_tokens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, L   = vis_tokens.shape
        valid  = vis_tokens != self.pad_token_id
        probs  = torch.rand(B, L, device=vis_tokens.device)
        mlm_mask   = (probs < self.mlm_mask_ratio) & valid
        mlm_labels = torch.full((B, L), -100, dtype=torch.long, device=vis_tokens.device)
        mlm_labels[mlm_mask] = vis_tokens[mlm_mask]
        mlm_tokens = vis_tokens.clone()
        mlm_tokens[mlm_mask] = self.mask_token_id
        return mlm_tokens, mlm_mask, mlm_labels

    # ------------------------------------------------------------------
    # Phase 1 — JEPA pretraining forward
    # ------------------------------------------------------------------

    def forward_pretrain(
        self,
        tokens: torch.Tensor,       # (B, L)
        target_mask: torch.Tensor,  # (B, L) True = target
    ) -> Dict[str, torch.Tensor]:
        B, L = tokens.shape

        vis_tokens, vis_positions, tgt_positions, tgt_padding, _ = \
            self._extract_visible_and_target(tokens, target_mask)

        mlm_tokens, mlm_mask, mlm_labels = self._apply_mlm_mask(vis_tokens)

        ctx_out       = self.context_encoder(mlm_tokens, vis_positions)
        context_cls   = ctx_out["cls"]
        context_tokens = ctx_out["tokens"]

        mlm_logits = self.mlm_head(context_tokens)

        with torch.no_grad():
            seq_positions = torch.arange(L, device=tokens.device).unsqueeze(0).expand(B, -1)
            tgt_out       = self.target_encoder(tokens, seq_positions)
            target_emb    = tgt_out["tokens"]  # (B, L, D)

        jepa_preds = self.predictor(
            context_emb=context_tokens,
            context_positions=vis_positions,
            target_positions=tgt_positions,
            target_padding_mask=tgt_padding,
        )

        tgt_pos_exp  = tgt_positions.unsqueeze(-1).expand(-1, -1, self.embed_dim)
        jepa_targets = torch.gather(target_emb, 1, tgt_pos_exp)

        return {
            "jepa_predictions":   jepa_preds,
            "jepa_targets":       jepa_targets.detach(),
            "jepa_target_padding": tgt_padding,
            "mlm_logits":         mlm_logits,
            "mlm_labels":         mlm_labels,
            "mlm_mask":           mlm_mask,
            "context_cls":        context_cls,
        }

    # ------------------------------------------------------------------
    # Phase 2 — Fine-tuning forward
    # ------------------------------------------------------------------

    def forward_finetune(
        self,
        tokens: torch.Tensor,        # (B, L)
        freeze_encoder: bool = False,
    ) -> torch.Tensor:               # (B,) logits
        B, L      = tokens.shape
        positions = torch.arange(L, device=tokens.device).unsqueeze(0).expand(B, -1)

        if freeze_encoder:
            with torch.no_grad():
                cls = self.context_encoder(tokens, positions)["cls"]
        else:
            cls = self.context_encoder(tokens, positions)["cls"]

        return self.cls_head(cls)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def encode(self, tokens: torch.Tensor, use_target: bool = True) -> torch.Tensor:
        """Extract CLS embeddings. use_target=True for stable inference embeddings."""
        B, L      = tokens.shape
        positions = torch.arange(L, device=tokens.device).unsqueeze(0).expand(B, -1)
        encoder   = self.target_encoder if use_target else self.context_encoder
        return encoder(tokens, positions)["cls"]


# =============================================================================
# 11. Loss Computation
# =============================================================================

def compute_pretrain_losses(
    out: Dict[str, torch.Tensor],
    model: PeptideJEPA,
    jepa_weight: float = 1.0,
    mlm_weight: float  = 0.5,
    sig_weight: float  = 10.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute JEPA + MLM + SIGReg losses for pretraining."""
    metrics = {}

    # JEPA loss (Smooth L1)
    pred    = out["jepa_predictions"]
    target  = out["jepa_targets"]
    tgt_pad = out["jepa_target_padding"]
    valid   = ~tgt_pad

    if valid.any():
        jepa_loss = F.smooth_l1_loss(pred[valid], target[valid])
        with torch.no_grad():
            jepa_cos = F.cosine_similarity(pred[valid], target[valid], dim=-1).mean().item()
    else:
        jepa_loss = torch.tensor(0.0, device=pred.device)
        jepa_cos  = 0.0
    metrics["jepa_loss"] = jepa_loss.item()
    metrics["jepa_cos"]  = jepa_cos

    # MLM loss (Cross-entropy on masked visible tokens)
    logits = out["mlm_logits"].reshape(-1, VOCAB_SIZE)
    labels = out["mlm_labels"].reshape(-1)
    mlm_loss = F.cross_entropy(logits, labels, ignore_index=-100)
    with torch.no_grad():
        mask_flat = out["mlm_mask"].reshape(-1)
        n_masked  = mask_flat.sum().clamp(min=1)
        mlm_acc   = ((logits.argmax(-1) == labels) & mask_flat).sum().float() / n_masked
    metrics["mlm_loss"] = mlm_loss.item()
    metrics["mlm_acc"]  = mlm_acc.item()

    # SIGReg (collapse prevention)
    sig_loss, sig_met = model.sigreg(out["context_cls"])
    metrics.update({f"sig_{k}": v for k, v in sig_met.items()})

    total = jepa_weight * jepa_loss + mlm_weight * mlm_loss + sig_weight * sig_loss
    metrics["loss"] = total.item()
    return total, metrics


# =============================================================================
# 12. Warmup + Cosine scheduler (per-step, not per-epoch)
# =============================================================================

def get_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.01,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear warmup then cosine decay, stepped every batch.

    warmup_steps: ramp lr from 0 → base_lr over this many batches
    total_steps:  total training batches (epochs × batches_per_epoch)
    min_lr_ratio: final lr = base_lr × min_lr_ratio
    """
    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return current_step / max(1, warmup_steps)
        progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine   = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(min_lr_ratio, cosine)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# =============================================================================
# 13. Training Loops
# =============================================================================

def pretrain_epoch(
    model: PeptideJEPA,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    device: torch.device,
    epoch: int,
    total_epochs: int,
    args,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
) -> Dict[str, float]:
    """One epoch of JEPA pretraining."""
    model.train()
    running = {}
    progress = epoch / max(total_epochs - 1, 1)
    model.set_ema_decay(progress)

    for batch in tqdm(loader, desc=f"Pretrain epoch {epoch+1}"):
        # Batch can be (tokens, labels) or just tokens — ignore labels
        tokens = batch[0] if isinstance(batch, (list, tuple)) else batch
        tokens = tokens.to(device)
        B, L   = tokens.shape

        # Generate target mask
        valid       = tokens != PAD_TOKEN_ID
        target_mask = multi_block_mask(
            seq_len=L,
            mask_ratio=model.jepa_mask_ratio,
            num_blocks=model.jepa_num_blocks,
            min_block_len=model.jepa_min_block_len,
            valid_mask=valid,
            device=device,
        )

        ctx = torch.cuda.amp.autocast() if scaler else contextlib_nullcontext()
        with ctx:
            out  = model.forward_pretrain(tokens, target_mask)
            loss, metrics = compute_pretrain_losses(
                out, model,
                jepa_weight=args.jepa_weight,
                mlm_weight=args.mlm_weight,
                sig_weight=args.sig_weight,
            )

        optimizer.zero_grad()
        if scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.update_ema()
        scheduler.step()  # per-batch step (warmup + cosine)

        for k, v in metrics.items():
            running[k] = running.get(k, 0.0) + v

    n = max(len(loader), 1)
    avg = {k: v / n for k, v in running.items()}
    avg["lr"] = scheduler.get_last_lr()[0]
    return avg


def finetune_epoch(
    model: PeptideJEPA,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    pos_weight: torch.Tensor,
    freeze_encoder: bool = False,
) -> Dict[str, float]:
    """One epoch of supervised fine-tuning."""
    model.train()
    if freeze_encoder:
        model.context_encoder.eval()
        for p in model.context_encoder.parameters():
            p.requires_grad = False

    total_loss, n_batches = 0.0, 0
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))

    for tokens, labels in tqdm(loader, desc="Fine-tune"):
        tokens = tokens.to(device)
        labels = labels.to(device)

        logits = model.forward_finetune(tokens, freeze_encoder=freeze_encoder)
        loss   = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches  += 1

    return {"loss": total_loss / max(n_batches, 1)}


@torch.no_grad()
def evaluate(
    model: PeptideJEPA,
    loader: DataLoader,
    device: torch.device,
    pos_weight: torch.Tensor,
) -> Dict[str, float]:
    """Evaluate on val or test set. Returns AUC, AUPRC, loss."""
    model.eval()
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    all_logits, all_labels = [], []

    for tokens, labels in loader:
        tokens = tokens.to(device)
        labels = labels.to(device)
        logits = model.forward_finetune(tokens, freeze_encoder=True)
        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())

    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)
    loss       = criterion(all_logits, all_labels).item()
    probs      = torch.sigmoid(all_logits).numpy()
    labels_np  = all_labels.numpy()

    metrics = {"loss": loss}
    if HAS_SKLEARN:
        metrics["auc"]   = roc_auc_score(labels_np, probs)
        metrics["auprc"] = average_precision_score(labels_np, probs)

    return metrics


# =============================================================================
# 13. Main
# =============================================================================

def get_args():
    p = argparse.ArgumentParser(description="P-JEPA for non-histone acetylation")
    p.add_argument("--phase", choices=["pretrain", "finetune", "both"], default="both")
    p.add_argument("--data-path",    default="../../../data/genomics/NHAC_deduplicated.csv",
                   help="Labeled NHAC CSV for fine-tuning (train/val/test split)")
    p.add_argument("--pretrain-csv", default=None,
                   help="Unlabeled CSV for pretraining (e.g. UniProt K-windows). "
                        "If not set, falls back to --data-path (NHAC only).")
    p.add_argument("--seq-col",     default="seq_31")
    p.add_argument("--output-dir",  default="outputs/nhac")
    p.add_argument("--checkpoint",  default=None, help="Path to pretrained checkpoint for fine-tuning")

    # Model — scaled up for 634K UniProt sequences
    p.add_argument("--embed-dim",       type=int,   default=256)
    p.add_argument("--num-layers",      type=int,   default=6)
    p.add_argument("--num-heads",       type=int,   default=8)
    p.add_argument("--ff-dim",          type=int,   default=512)
    p.add_argument("--predictor-dim",   type=int,   default=128)
    p.add_argument("--predictor-depth", type=int,   default=3)
    p.add_argument("--var-gamma",       type=float, default=1.0)

    # Pretraining
    p.add_argument("--pretrain-epochs", type=int,   default=50)
    p.add_argument("--pretrain-lr",     type=float, default=3e-5)
    p.add_argument("--warmup-steps",    type=int,   default=500,
                   help="Linear warmup steps before cosine decay (per-batch)")
    p.add_argument("--jepa-weight",     type=float, default=1.0)
    p.add_argument("--mlm-weight",      type=float, default=0.5)
    p.add_argument("--sig-weight",      type=float, default=10.0)

    # Fine-tuning
    p.add_argument("--finetune-epochs", type=int,   default=30)
    p.add_argument("--finetune-lr",     type=float, default=3e-4)
    p.add_argument("--freeze-encoder",  action="store_true",
                   help="Freeze encoder during fine-tuning (only train cls head)")
    p.add_argument("--label-fraction",  type=float, default=1.0,
                   help="Fraction of training labels to use (for low-data experiments)")

    # Training
    p.add_argument("--batch-size",  type=int, default=64)
    p.add_argument("--device",      default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--wandb",       action="store_true")
    p.add_argument("--run-name",    default="pjepa-nhac")
    p.add_argument("--log-file",    default=None,
                   help="Path to log file (default: <output-dir>/train.log)")
    return p.parse_args()


# =============================================================================
# Logger — writes to stdout AND a log file simultaneously
# =============================================================================

class Logger:
    """Thin wrapper: log(msg) prints to terminal and appends to a file."""

    def __init__(self, log_path: str):
        self.log_path = log_path
        # Write header with timestamp
        import datetime
        with open(log_path, "w") as f:
            f.write(f"# P-JEPA training log — {datetime.datetime.now()}\n\n")

    def log(self, msg: str = ""):
        print(msg)
        with open(self.log_path, "a") as f:
            f.write(msg + "\n")


def main():
    args   = get_args()
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    log_path = args.log_file or os.path.join(args.output_dir, "train.log")
    logger   = Logger(log_path)

    logger.log(f"Device: {device}")
    logger.log(f"Loading data from {args.data_path} ...")
    train_ds, val_ds, test_ds = load_nhac(args.data_path, seq_col=args.seq_col)

    # Low-data experiment: subsample training labels
    if args.label_fraction < 1.0:
        n_total = len(train_ds)
        n_keep  = max(1, int(n_total * args.label_fraction))
        indices = random.sample(range(n_total), n_keep)
        train_ds.sequences = [train_ds.sequences[i] for i in indices]
        train_ds.labels    = [train_ds.labels[i]    for i in indices]
        logger.log(f"Low-data mode: using {n_keep}/{n_total} training samples "
                   f"({args.label_fraction*100:.0f}%)")

    pos_weight = compute_pos_weight(train_ds)
    logger.log(f"Class balance — pos: {sum(train_ds.labels)}, neg: {len(train_ds)-sum(train_ds.labels)}")
    logger.log(f"Positive weight for BCE: {pos_weight.item():.2f}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Build model
    model = PeptideJEPA(
        embed_dim=args.embed_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        ff_dim=args.ff_dim,
        predictor_dim=args.predictor_dim,
        predictor_depth=args.predictor_depth,
        var_gamma=args.var_gamma,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    logger.log(f"Model parameters: {n_params:,}")
    logger.log(f"Args: {vars(args)}")
    logger.log("")

    if args.wandb and HAS_WANDB:
        wandb.init(project="pjepa-nhac", name=args.run_name, config=vars(args))

    # -----------------------------------------------------------------------
    # Phase 1: Pretraining
    # -----------------------------------------------------------------------
    if args.phase in ("pretrain", "both"):
        logger.log("=== Phase 1: JEPA Pretraining ===")

        # Use UniProt K-windows if provided, otherwise fall back to NHAC sequences
        pretrain_path = args.pretrain_csv or args.data_path
        all_df        = pd.read_csv(pretrain_path).reset_index(drop=True)
        pretrain_ds   = NHACDataset(all_df, seq_col=args.seq_col, label_col=None)
        pretrain_loader = DataLoader(pretrain_ds, batch_size=args.batch_size,
                                     shuffle=True, num_workers=0)
        source = "UniProt K-windows" if args.pretrain_csv else "NHAC sequences (no labels)"
        logger.log(f"Pretrain corpus: {len(pretrain_ds):,} sequences  [{source}]")
        logger.log(f"Source file:     {pretrain_path}")

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.pretrain_lr,
            betas=(0.9, 0.98), weight_decay=0.05,
        )
        total_steps = args.pretrain_epochs * len(pretrain_loader)
        scheduler   = get_warmup_cosine_scheduler(
            optimizer,
            warmup_steps=args.warmup_steps,
            total_steps=total_steps,
        )
        logger.log(f"Scheduler: warmup {args.warmup_steps} steps → cosine over {total_steps:,} total steps")
        logger.log(f"LR: {args.pretrain_lr:.2e}  (effective after warmup)")

        best_loss = float("inf")
        for epoch in range(args.pretrain_epochs):
            metrics = pretrain_epoch(
                model, pretrain_loader, optimizer, scheduler,
                device, epoch, args.pretrain_epochs, args,
            )
            # scheduler is stepped per-batch inside pretrain_epoch — no step() here

            # Weighted contributions show what's actually driving the total loss
            w_jepa = args.jepa_weight * metrics.get('jepa_loss', 0)
            w_mlm  = args.mlm_weight  * metrics.get('mlm_loss',  0)
            w_sig  = args.sig_weight  * metrics.get('sig_sigreg', 0)

            line1 = (f"[Pretrain {epoch+1:3d}/{args.pretrain_epochs}] "
                     f"total={metrics.get('loss', 0):.4f}  lr={metrics.get('lr', 0):.2e}  |  "
                     f"jepa={metrics.get('jepa_loss', 0):.4f}(x{args.jepa_weight})={w_jepa:.4f}  "
                     f"mlm={metrics.get('mlm_loss', 0):.4f}(x{args.mlm_weight})={w_mlm:.4f}  "
                     f"sig={metrics.get('sig_sigreg', 0):.4f}(x{args.sig_weight})={w_sig:.4f}")
            line2 = (f"{'':>15}"
                     f"jepa_cos={metrics.get('jepa_cos', 0):.3f}  "
                     f"mlm_acc={metrics.get('mlm_acc', 0):.3f}  "
                     f"emb_std={metrics.get('sig_std_mean', 0):.3f}  "
                     f"var_floor={metrics.get('sig_var_floor', 0):.4f}")
            logger.log(line1)
            logger.log(line2)

            if args.wandb and HAS_WANDB:
                wandb.log({"pretrain/" + k: v for k, v in metrics.items()}, step=epoch)

            if metrics.get("loss", float("inf")) < best_loss:
                best_loss = metrics["loss"]
                ckpt_path = os.path.join(args.output_dir, "pretrain_best.pt")
                torch.save({
                    "epoch":            epoch,
                    "model_state_dict": model.state_dict(),
                    "config": {
                        "embed_dim":       args.embed_dim,
                        "num_layers":      args.num_layers,
                        "num_heads":       args.num_heads,
                        "ff_dim":          args.ff_dim,
                        "predictor_dim":   args.predictor_dim,
                        "predictor_depth": args.predictor_depth,
                        "var_gamma":       args.var_gamma,
                    },
                }, ckpt_path)

        logger.log(f"Pretraining done. Best loss={best_loss:.4f}  checkpoint: {ckpt_path}")
        logger.log("")

    # -----------------------------------------------------------------------
    # Phase 2: Fine-tuning
    # -----------------------------------------------------------------------
    if args.phase in ("finetune", "both"):
        logger.log("=== Phase 2: Fine-tuning ===")

        # Load pretrained checkpoint if provided
        ckpt_path = args.checkpoint or os.path.join(args.output_dir, "pretrain_best.pt")
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"], strict=False)
            logger.log(f"Loaded pretrained weights from {ckpt_path}")
        else:
            logger.log("No pretrained checkpoint found — fine-tuning from scratch (ablation mode)")

        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.finetune_lr, weight_decay=0.01,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.finetune_epochs, eta_min=1e-6,
        )

        best_auc   = 0.0
        best_epoch = 0
        ft_ckpt    = os.path.join(args.output_dir, "finetune_best.pt")
        for epoch in range(args.finetune_epochs):
            train_met = finetune_epoch(
                model, train_loader, optimizer, device,
                pos_weight, freeze_encoder=args.freeze_encoder,
            )
            val_met = evaluate(model, val_loader, device, pos_weight)
            scheduler.step()

            auc   = val_met.get("auc",   0.0)
            auprc = val_met.get("auprc", 0.0)
            line = (f"[Finetune {epoch+1:3d}/{args.finetune_epochs}] "
                    f"train_loss={train_met['loss']:.4f}  "
                    f"val_loss={val_met['loss']:.4f}  "
                    f"val_AUC={auc:.4f}  val_AUPRC={auprc:.4f}")
            logger.log(line)

            if args.wandb and HAS_WANDB:
                wandb.log({
                    "finetune/train_loss": train_met["loss"],
                    **{"finetune/val_" + k: v for k, v in val_met.items()},
                }, step=epoch)

            if auc > best_auc:
                best_auc   = auc
                best_epoch = epoch + 1
                torch.save({"epoch": epoch, "model_state_dict": model.state_dict()}, ft_ckpt)

        logger.log(f"\nBest val AUC: {best_auc:.4f} at epoch {best_epoch}")

        # Final test evaluation
        model.load_state_dict(torch.load(ft_ckpt, map_location=device,
                                         weights_only=False)["model_state_dict"])
        test_met = evaluate(model, test_loader, device, pos_weight)
        logger.log("")
        logger.log("=== Test Results ===")
        logger.log(f"  AUC:   {test_met.get('auc',   0.0):.4f}  (TransPTM baseline: 0.83)")
        logger.log(f"  AUPRC: {test_met.get('auprc', 0.0):.4f}  (TransPTM baseline: 0.51)")
        logger.log(f"  Loss:  {test_met['loss']:.4f}")
        logger.log(f"\nLog saved to: {log_path}")

        if args.wandb and HAS_WANDB:
            wandb.log({"test/" + k: v for k, v in test_met.items()})


if __name__ == "__main__":
    main()
