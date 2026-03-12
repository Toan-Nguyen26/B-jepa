#!/usr/bin/env python3
"""B-JEPA v6.0 — 

A genuine Joint-Embedding Predictive Architecture where JEPA is the PRIMARY
learning signal and MLM serves as an anti-collapse guard.

Why v6 exists (v5.0 post-mortem):
  v5.0 was a good MLM model with JEPA decorations. The "CLS predictor" was an
  MLP denoiser on a single vector (self-attention on length 1 = identity).
  20-30% masking created no information gap. JEPA gradients never reached
  per-token representations. Zeroing JEPA weight would not change outcomes.

v6 description :
  1. ASYMMETRIC MASKING: Context sees only 30-50% of tokens (50-70% masked).
     High mask ratio forces abstract representations — can't memorize.
  2. POSITIONAL CROSS-ATTENTION PREDICTOR: Learnable target position queries
     cross-attend to visible context embeddings. The predictor never sees the
     full sequence layout, preventing trivial neighbor interpolation.
  3. JEPA AS PRIMARY LOSS (weight=5.0): Smooth-L1 on per-token latent
     predictions at masked positions. This is the dominant learning signal.
  4. MLM AS ANTI-COLLAPSE GUARD (weight=0.5): 15% random masking within
     visible tokens only. Prevents the collapse that killed v4.5 by forcing
     different tokens to produce different hidden states.
  5. SIGReg REPLACES VICReg: One hyperparameter, provably optimal Gaussian
     target distribution (LeJEPA, Balestriero & LeCun 2025).
  6. PROPER ENCODER: RMSNorm, RoPE, SwiGLU, QK-Norm (v4.0 architecture),
     not vanilla nn.TransformerEncoderLayer.

Why this won't collapse like v4.5:
  v4.5 collapsed because the predictor saw the FULL sequence (all L positions)
  with mask tokens at target positions. DNA's local redundancy meant neighbors
  trivially leaked the answer. v6 fixes this: the context encoder processes
  ONLY visible tokens. The predictor receives compressed context embeddings
  and must cross-attend from positional queries — no neighbor interpolation.
  Additionally, MLM on visible tokens provides a structural anti-collapse
  guarantee that v4.5 lacked entirely.

Data:     pretrain_2M.csv (2M fragments, 6326 genomes), BPE vocab=4096
Encoder:  12L x 576D x 9H, RoPE, RMSNorm, SwiGLU, QK-Norm, [CLS] token
Target:   EMA copy (tau = 0.996 -> 1.0 cosine)
Predictor: 6L x 384D x 6H, cross-attention from target queries to context KV
Masking:  Multi-block, 4 target blocks, curriculum 50% -> 70%
MLM:      15% random within visible tokens (anti-collapse only)

Loss = 5.0 * JEPA_smooth_l1 + 0.5 * MLM_ce + 10.0 * SIGReg + 1.0 * GC_adv

References:
  [1] Assran et al. "I-JEPA." CVPR 2023.
  [2] Balestriero & LeCun. "LeJEPA." arXiv:2511.08544, Nov 2025.
  [3] Larey et al. "JEPA-DNA." arXiv:2602.17162, Feb 2026.
  [4] Mo et al. "C-JEPA." NeurIPS 2024 Spotlight.

Usage:
  python bdna_jepa/models/jepa_v5/pretrain_v6.py \\
      --data-path data/processed/pretrain_2M.csv \\
      --tokenizer-path data/tokenizer/bpe_4096.json \\
      --epochs 30 --batch-size 64 --lr 3e-4
"""
from __future__ import annotations

import argparse
import copy
import math
import os
import random
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kw):
        return iterable

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


# =============================================================================
# 1. Building Blocks — RMSNorm, RoPE, SwiGLU, MultiHeadAttention
# =============================================================================

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (Zhang & Sennrich, 2019)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).type_as(x) * self.weight


class RotaryEmbedding(nn.Module):
    """Rotary Positional Embedding (Su et al., 2021)."""

    def __init__(self, dim: int, max_seq_len: int = 4096, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, positions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """positions: (L,) integer position IDs -> (L, head_dim) cos/sin."""
        max_pos = positions.max().item() + 1
        if max_pos > self.cos_cached.size(0):
            self._build_cache(max_pos * 2)
        return self.cos_cached[positions], self.sin_cached[positions]


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """q, k: (B, H, L, D), cos/sin: (L, D) or (B, L, D)."""
    if cos.dim() == 2:
        cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, L, D)
        sin = sin.unsqueeze(0).unsqueeze(0)
    elif cos.dim() == 3:
        cos = cos.unsqueeze(1)  # (B, 1, L, D)
        sin = sin.unsqueeze(1)
    q_rot = q * cos + rotate_half(q) * sin
    k_rot = k * cos + rotate_half(k) * sin
    return q_rot, k_rot


class SwiGLU(nn.Module):
    """SwiGLU feedforward (Shazeer, 2020). Used by Llama, PaLM."""

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
    """Multi-head attention with RoPE and QK-Norm. Supports variable positions."""

    def __init__(self, dim: int, num_heads: int, qk_norm: bool = True, bias: bool = False):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=bias)
        self.out_proj = nn.Linear(dim, dim, bias=bias)
        self.qk_norm = qk_norm
        if qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)

    def forward(
        self, x: torch.Tensor,
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, L, D = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(2)
        q = q.transpose(1, 2)  # (B, H, L, D)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        if rope_cos is not None:
            q, k = apply_rotary_pos_emb(q, k, rope_cos, rope_sin)

        attn_mask = None
        if key_padding_mask is not None:
            # key_padding_mask: (B, L) True = IGNORE
            attn_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, L)
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
        self.head_dim = dim // num_heads
        self.q_proj = nn.Linear(dim, dim, bias=bias)
        self.kv_proj = nn.Linear(dim, 2 * dim, bias=bias)
        self.out_proj = nn.Linear(dim, dim, bias=bias)
        self.qk_norm = qk_norm
        if qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)

    def forward(
        self, queries: torch.Tensor, context: torch.Tensor,
        q_rope_cos: Optional[torch.Tensor] = None,
        q_rope_sin: Optional[torch.Tensor] = None,
        kv_rope_cos: Optional[torch.Tensor] = None,
        kv_rope_sin: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, Lq, D = queries.shape
        Lk = context.shape[1]

        q = self.q_proj(queries).reshape(B, Lq, self.num_heads, self.head_dim).transpose(1, 2)
        kv = self.kv_proj(context).reshape(B, Lk, 2, self.num_heads, self.head_dim)
        k, v = kv.unbind(2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        if q_rope_cos is not None:
            q = q * q_rope_cos.unsqueeze(1) + rotate_half(q) * q_rope_sin.unsqueeze(1)
        if kv_rope_cos is not None:
            k = k * kv_rope_cos.unsqueeze(1) + rotate_half(k) * kv_rope_sin.unsqueeze(1)

        out = F.scaled_dot_product_attention(q, k, v)
        return self.out_proj(out.transpose(1, 2).reshape(B, Lq, D))


class TransformerBlock(nn.Module):
    """Pre-norm transformer block with RMSNorm + SwiGLU."""

    def __init__(self, dim: int, num_heads: int, ff_dim: int,
                 qk_norm: bool = True, bias: bool = False):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = MultiHeadAttention(dim, num_heads, qk_norm=qk_norm, bias=bias)
        self.norm2 = RMSNorm(dim)
        self.ffn = SwiGLU(dim, ff_dim, bias=bias)

    def forward(self, x, rope_cos=None, rope_sin=None, key_padding_mask=None):
        x = x + self.attn(self.norm1(x), rope_cos, rope_sin, key_padding_mask)
        x = x + self.ffn(self.norm2(x))
        return x


# =============================================================================
# 2. Context Encoder — Processes ONLY visible tokens (true I-JEPA design)
# =============================================================================

class ContextEncoder(nn.Module):
    """Transformer encoder that processes ONLY visible (unmasked) tokens.

    Key difference from v5.0: the encoder never sees the full sequence.
    Visible tokens are extracted, encoded with their ORIGINAL position IDs
    (via RoPE), so the encoder knows where each token was in the sequence
    but cannot interpolate across masked gaps.

    Prepends a [CLS] token at position 0 for pooled representation.
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
        qk_norm: bool = True,
        bias: bool = False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        self.pad_token_id = pad_token_id
        self.head_dim = embed_dim // num_heads

        self.token_emb = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_token_id)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.embed_dropout = nn.Dropout(dropout)

        # RoPE for positional encoding (applied per original position)
        self.rope = RotaryEmbedding(self.head_dim, max_seq_len + 1)

        self.layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim, qk_norm=qk_norm, bias=bias)
            for _ in range(num_layers)
        ])
        self.final_norm = RMSNorm(embed_dim)
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.token_emb.weight, std=0.02)
        nn.init.normal_(self.cls_token, std=0.02)
        if self.token_emb.padding_idx is not None:
            self.token_emb.weight.data[self.token_emb.padding_idx].zero_()
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        tokens: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            tokens: (B, L_vis) visible token IDs (variable length, padded)
            position_ids: (B, L_vis) original position IDs for each visible token

        Returns:
            cls: (B, D) CLS embedding
            tokens: (B, L_vis, D) per-token embeddings (excluding CLS)
        """
        B, L = tokens.shape

        x = self.token_emb(tokens)  # (B, L, D)

        # Prepend CLS
        cls_expanded = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_expanded, x], dim=1)  # (B, L+1, D)
        x = self.embed_dropout(x)

        # RoPE positions: CLS gets position 0, visible tokens get their originals + 1
        cls_pos = torch.zeros(B, 1, device=tokens.device, dtype=torch.long)
        rope_positions = torch.cat([cls_pos, position_ids + 1], dim=1)  # (B, L+1)

        # Build per-sample RoPE cos/sin: (B, L+1, head_dim)
        rope_cos_list, rope_sin_list = [], []
        for b in range(B):
            c, s = self.rope(rope_positions[b])
            rope_cos_list.append(c)
            rope_sin_list.append(s)
        rope_cos = torch.stack(rope_cos_list, dim=0)  # (B, L+1, head_dim)
        rope_sin = torch.stack(rope_sin_list, dim=0)

        # Padding mask for attention: pad tokens should be ignored
        pad_mask = (tokens == self.pad_token_id)
        cls_mask = torch.zeros(B, 1, device=tokens.device, dtype=torch.bool)
        key_padding_mask = torch.cat([cls_mask, pad_mask], dim=1)  # (B, L+1), True=ignore

        for layer in self.layers:
            x = layer(x, rope_cos, rope_sin, key_padding_mask)

        x = self.final_norm(x)

        return {
            "cls": x[:, 0, :],       # (B, D)
            "tokens": x[:, 1:, :],   # (B, L_vis, D)
        }


# =============================================================================
# 3. Target Encoder — Processes ALL tokens (standard)
# =============================================================================

class TargetEncoder(nn.Module):
    """Full-sequence encoder for producing target embeddings.

    Same architecture as ContextEncoder but always sees the complete sequence.
    Frozen — updated only via EMA from context encoder.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 576,
        num_layers: int = 12,
        num_heads: int = 9,
        ff_dim: int = 2304,
        max_seq_len: int = 512,
        dropout: float = 0.0,  # No dropout for target
        pad_token_id: int = 0,
        qk_norm: bool = True,
        bias: bool = False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.pad_token_id = pad_token_id
        self.head_dim = embed_dim // num_heads

        self.token_emb = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_token_id)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

        self.rope = RotaryEmbedding(self.head_dim, max_seq_len + 1)

        self.layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim, qk_norm=qk_norm, bias=bias)
            for _ in range(num_layers)
        ])
        self.final_norm = RMSNorm(embed_dim)

    @torch.no_grad()
    def forward(self, tokens: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            tokens: (B, L) full token IDs

        Returns:
            cls: (B, D)
            tokens: (B, L, D) per-token embeddings (excluding CLS)
        """
        B, L = tokens.shape
        x = self.token_emb(tokens)
        cls_expanded = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_expanded, x], dim=1)  # (B, L+1, D)

        # Sequential positions: CLS=0, tokens=1..L
        positions = torch.arange(L + 1, device=tokens.device)
        rope_cos, rope_sin = self.rope(positions)  # (L+1, head_dim)

        pad_mask = (tokens == self.pad_token_id)
        cls_mask = torch.zeros(B, 1, device=tokens.device, dtype=torch.bool)
        key_padding_mask = torch.cat([cls_mask, pad_mask], dim=1)

        for layer in self.layers:
            x = layer(x, rope_cos, rope_sin, key_padding_mask)

        x = self.final_norm(x)
        return {
            "cls": x[:, 0, :],
            "tokens": x[:, 1:, :],  # (B, L, D)
        }


# =============================================================================
# 4. JEPA Predictor — Cross-attention from positional queries to context
# =============================================================================

class JEPAPredictor(nn.Module):
    """True I-JEPA predictor with cross-attention.

    Target positions are represented as learnable mask tokens + positional
    embeddings. These queries cross-attend to the visible context embeddings.
    The predictor NEVER sees the full sequence layout — only compressed context
    and positional queries for where to predict.

    Architecture:
        target_queries = mask_token + pos_embed[target_positions]
        for each block:
            target_queries = cross_attn(target_queries, context_embeddings)
            target_queries = self_attn(target_queries)
            target_queries = FFN(target_queries)
        output = project(target_queries)  # -> D_enc predictions
    """

    def __init__(
        self,
        encoder_dim: int = 576,
        predictor_dim: int = 384,
        depth: int = 6,
        num_heads: int = 6,
        max_seq_len: int = 512,
        qk_norm: bool = True,
        bias: bool = False,
    ):
        super().__init__()
        self.predictor_dim = predictor_dim

        # Project context from encoder dim to predictor dim
        self.context_proj = nn.Sequential(
            nn.Linear(encoder_dim, predictor_dim, bias=bias),
            RMSNorm(predictor_dim),
        )

        # Learnable mask token for target queries
        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        # Positional embedding for target query positions (original sequence positions)
        self.pos_embed = nn.Embedding(max_seq_len + 1, predictor_dim)
        nn.init.trunc_normal_(self.pos_embed.weight, std=0.02)

        # Positional embedding for context positions (for cross-attention keys)
        self.context_pos_embed = nn.Embedding(max_seq_len + 1, predictor_dim)
        nn.init.trunc_normal_(self.context_pos_embed.weight, std=0.02)

        # Predictor blocks: cross-attn -> self-attn -> FFN
        self.blocks = nn.ModuleList()
        for _ in range(depth):
            self.blocks.append(nn.ModuleDict({
                "cross_norm_q": RMSNorm(predictor_dim),
                "cross_norm_kv": RMSNorm(predictor_dim),
                "cross_attn": CrossAttention(predictor_dim, num_heads, qk_norm=qk_norm, bias=bias),
                "self_norm": RMSNorm(predictor_dim),
                "self_attn": MultiHeadAttention(predictor_dim, num_heads, qk_norm=qk_norm, bias=bias),
                "ffn_norm": RMSNorm(predictor_dim),
                "ffn": SwiGLU(predictor_dim, predictor_dim * 2, bias=bias),
            }))

        self.final_norm = RMSNorm(predictor_dim)
        self.output_proj = nn.Linear(predictor_dim, encoder_dim, bias=bias)

    def forward(
        self,
        context_emb: torch.Tensor,
        context_positions: torch.Tensor,
        target_positions: torch.Tensor,
        target_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            context_emb: (B, L_ctx, D_enc) visible token embeddings from context encoder
            context_positions: (B, L_ctx) original position IDs of context tokens
            target_positions: (B, L_tgt) original position IDs of target tokens
            target_padding_mask: (B, L_tgt) True = padding (ignore)

        Returns:
            predictions: (B, L_tgt, D_enc) predicted target embeddings
        """
        B, L_tgt = target_positions.shape

        # Project context to predictor dim and add positional info
        ctx = self.context_proj(context_emb)  # (B, L_ctx, D_pred)
        ctx = ctx + self.context_pos_embed(context_positions)

        # Build target queries: mask_token + positional embedding
        queries = self.mask_token.expand(B, L_tgt, -1) + self.pos_embed(target_positions)

        # Apply predictor blocks
        for block in self.blocks:
            # Cross-attention: target queries attend to context
            queries = queries + block["cross_attn"](
                block["cross_norm_q"](queries),
                block["cross_norm_kv"](ctx),
            )
            # Self-attention among target queries
            queries = queries + block["self_attn"](block["self_norm"](queries))
            # FFN
            queries = queries + block["ffn"](block["ffn_norm"](queries))

        return self.output_proj(self.final_norm(queries))  # (B, L_tgt, D_enc)


# =============================================================================
# 5. MLM Head (anti-collapse guard)
# =============================================================================

class MLMHead(nn.Module):
    """Masked Language Model head: RMSNorm -> Dense -> SiLU -> Linear(vocab)."""

    def __init__(self, embed_dim: int, vocab_size: int):
        super().__init__()
        self.norm = RMSNorm(embed_dim)
        self.dense = nn.Linear(embed_dim, embed_dim)
        self.act = nn.SiLU()
        self.output = nn.Linear(embed_dim, vocab_size)
        nn.init.normal_(self.dense.weight, std=0.02)
        nn.init.normal_(self.output.weight, std=0.02)

    def forward(self, token_embeddings: torch.Tensor) -> torch.Tensor:
        """(B, L, D) -> (B, L, vocab_size)"""
        return self.output(self.act(self.dense(self.norm(token_embeddings))))


# =============================================================================
# 6. SIGReg — Sketched Isotropic Gaussian Regularization (LeJEPA)
# =============================================================================

class SIGReg(nn.Module):
    """SIGReg from LeJEPA (Balestriero & LeCun, 2025) + variance floor.

    Projects embeddings onto K random 1D directions, tests each marginal
    against N(0,1) via Epps-Pulley characteristic function test.
    Loss = 0 when distribution is perfectly isotropic Gaussian.

    Added: VICReg-style variance floor (hinge on projection stds) to prevent
    scale collapse. The original SIGReg standardizes before testing, making it
    blind to embeddings shrinking toward zero while maintaining Gaussian shape.
    The var_floor term penalizes projection stds below gamma, ensuring the
    N(0,1) target is enforced for both shape AND scale.
    """

    def __init__(self, num_slices: int = 512, num_points: int = 17,
                 var_gamma: float = 1.0):
        super().__init__()
        self.num_slices = num_slices
        self.var_gamma = var_gamma
        t_max = 2.0
        t_points = torch.linspace(0, t_max, num_points + 1)[1:]
        self.register_buffer('t_points', t_points)
        dt = t_max / num_points
        weights = torch.full((num_points,), dt)
        weights[0] = dt / 2
        weights[-1] = dt / 2
        self.register_buffer('weights', weights)

    def forward(self, embeddings: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            embeddings: (B, D) batch of embeddings

        Returns:
            loss: scalar
            metrics: dict with diagnostic values
        """
        B, D = embeddings.shape
        if B < 4:
            zero = torch.tensor(0.0, device=embeddings.device, requires_grad=True)
            return zero, {"sigreg": 0.0, "std_mean": 0.0, "var_floor": 0.0}

        z = embeddings.float()

        # Random projection directions
        directions = torch.randn(D, self.num_slices, device=z.device, dtype=z.dtype)
        directions = F.normalize(directions, dim=0)
        proj = z @ directions  # (B, K)

        # Projection stds (before standardization)
        std = proj.std(dim=0)

        # Variance floor: penalize projections with std < gamma
        # Prevents scale collapse that the Gaussianity test misses
        var_floor = F.relu(self.var_gamma - std).mean()

        # Standardize for Gaussianity test
        proj = (proj - proj.mean(dim=0, keepdim=True)) / (std + 1e-8)

        # Epps-Pulley test: compare empirical CF to N(0,1) CF
        total = torch.tensor(0.0, device=z.device)
        for t_idx, t in enumerate(self.t_points):
            cos_tp = torch.cos(t * proj)
            sin_tp = torch.sin(t * proj)
            ecf_real = cos_tp.mean(dim=0)
            ecf_imag = sin_tp.mean(dim=0)
            ecf_sq = ecf_real ** 2 + ecf_imag ** 2
            tcf = math.exp(-0.5 * t.item() ** 2)
            integrand = ecf_sq - 2 * ecf_real * tcf + tcf ** 2
            total = total + self.weights[t_idx] * integrand.mean()

        loss = total + var_floor

        metrics = {
            "sigreg": total.item(),
            "var_floor": var_floor.item(),
            "std_mean": std.mean().item(),
        }
        return loss, metrics


# =============================================================================
# 7. GC Adversary (gradient reversal for debiasing)
# =============================================================================

class _GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.clone()

    @staticmethod
    def backward(ctx, grad):
        return -ctx.lam * grad, None


class GCAdversary(nn.Module):
    """Gradient-reversal adversary for GC content debiasing."""

    def __init__(self, embed_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, emb: torch.Tensor, lam: float = 1.0) -> torch.Tensor:
        return self.net(_GradReverse.apply(emb, lam)).squeeze(-1)

    @staticmethod
    def ganin_lambda(epoch: int, total: int, gamma: float = 10.0) -> float:
        p = epoch / max(total - 1, 1)
        return 2.0 / (1.0 + math.exp(-gamma * p)) - 1.0


# =============================================================================
# 8. Multi-Block Masking (I-JEPA style) with Curriculum
# =============================================================================

def multi_block_mask(
    seq_len: int,
    mask_ratio: float,
    num_blocks: int,
    min_block_len: int,
    valid_mask: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Generate I-JEPA multi-block target masks.

    Creates num_blocks non-overlapping contiguous spans covering ~mask_ratio
    of valid tokens. Returns (B, L) bool where True = target (masked).

    Vectorized outer loop, per-sample inner placement.
    """
    B = valid_mask.shape[0]
    target_mask = torch.zeros(B, seq_len, dtype=torch.bool, device=device)
    valid_lens = valid_mask.sum(dim=1)

    for b in range(B):
        L_valid = valid_lens[b].item()
        n_target = int(L_valid * mask_ratio)
        if L_valid < min_block_len * 2 or n_target < min_block_len:
            blen = max(1, min(min_block_len, int(L_valid) - 1))
            if blen > 0 and int(L_valid) > blen:
                start = random.randint(0, int(L_valid) - blen)
                target_mask[b, start:start + blen] = True
            continue

        per_block = max(min_block_len, n_target // num_blocks)
        remaining = n_target
        occupied = torch.zeros(int(L_valid), dtype=torch.bool)

        for _ in range(num_blocks):
            if remaining < min_block_len:
                break
            blen = min(max(min_block_len, per_block), remaining, int(L_valid))

            placed = False
            for _attempt in range(50):
                max_start = int(L_valid) - blen
                if max_start < 0:
                    break
                start = random.randint(0, max_start)
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
                    actual = min(blen, len(free), int(L_valid) - start)
                    actual = max(1, actual)
                    occupied[start:start + actual] = True
                    target_mask[b, start:start + actual] = True
                    remaining -= actual

    target_mask &= valid_mask
    return target_mask


def curriculum_schedule(epoch: int, total: int, start: float, end: float) -> float:
    """Cosine curriculum ramp from start -> end."""
    p = epoch / max(total - 1, 1)
    t = 0.5 * (1 - math.cos(math.pi * p))
    return start + t * (end - start)


# =============================================================================
# 9. B-JEPA v6 Model — True JEPA + MLM Anti-Collapse
# =============================================================================

class BJEPAv6(nn.Module):
    """True JEPA for bacterial DNA.

    Forward pass:
      1. Multi-block masking: select 50-70% as target, 30-50% as context
      2. Extract visible tokens with original position IDs
      3. Within visible tokens, randomly mask 15% for MLM (anti-collapse)
      4. Context encoder: visible tokens (with MLM masks) -> per-token + CLS
      5. MLM head: predict 15%-masked visible tokens (anti-collapse loss)
      6. Target encoder (EMA): ALL tokens -> per-token target embeddings
      7. JEPA predictor: cross-attend from target positions to context -> predictions
      8. JEPA loss: smooth_l1(predictions, target embeddings at masked positions)
      9. SIGReg on context CLS (collapse prevention)
      10. GC adversary on CLS (debiasing)
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 576,
        num_layers: int = 12,
        num_heads: int = 9,
        ff_dim: int = 2304,
        max_seq_len: int = 512,
        predictor_dim: int = 384,
        predictor_depth: int = 6,
        predictor_heads: int = 6,
        pad_token_id: int = 0,
        mask_token_id: int = 1,
        ema_start: float = 0.996,
        ema_end: float = 1.0,
        mlm_mask_ratio: float = 0.15,
        var_gamma: float = 1.0,
        qk_norm: bool = True,
        bias: bool = False,
    ):
        super().__init__()
        self.pad_token_id = pad_token_id
        self.mask_token_id = mask_token_id
        self.embed_dim = embed_dim
        self.ema_start = ema_start
        self.ema_end = ema_end
        self._ema_decay = ema_start
        self.mlm_mask_ratio = mlm_mask_ratio
        self.max_seq_len = max_seq_len

        # Context encoder (trainable) — processes ONLY visible tokens
        self.context_encoder = ContextEncoder(
            vocab_size=vocab_size, embed_dim=embed_dim,
            num_layers=num_layers, num_heads=num_heads,
            ff_dim=ff_dim, max_seq_len=max_seq_len,
            pad_token_id=pad_token_id, qk_norm=qk_norm, bias=bias,
        )

        # Target encoder (frozen, EMA) — processes ALL tokens
        self.target_encoder = TargetEncoder(
            vocab_size=vocab_size, embed_dim=embed_dim,
            num_layers=num_layers, num_heads=num_heads,
            ff_dim=ff_dim, max_seq_len=max_seq_len,
            pad_token_id=pad_token_id, qk_norm=qk_norm, bias=bias,
        )

        # Initialize target from context
        self._sync_target_from_context()
        for p in self.target_encoder.parameters():
            p.requires_grad = False

        # JEPA predictor: cross-attention from target positions to context
        self.predictor = JEPAPredictor(
            encoder_dim=embed_dim, predictor_dim=predictor_dim,
            depth=predictor_depth, num_heads=predictor_heads,
            max_seq_len=max_seq_len, qk_norm=qk_norm, bias=bias,
        )

        # MLM head (anti-collapse)
        self.mlm_head = MLMHead(embed_dim, vocab_size)

        # SIGReg (replaces VICReg) + variance floor
        self.sigreg = SIGReg(num_slices=512, num_points=17, var_gamma=var_gamma)

        # GC adversary
        self.gc_adversary = GCAdversary(embed_dim, hidden_dim=64)

    def _sync_target_from_context(self):
        """Copy context encoder weights to target encoder."""
        ctx_state = self.context_encoder.state_dict()
        tgt_state = self.target_encoder.state_dict()
        for key in tgt_state:
            if key in ctx_state:
                tgt_state[key].copy_(ctx_state[key])

    def set_ema_decay(self, progress: float) -> float:
        t0, t1 = self.ema_start, self.ema_end
        self._ema_decay = t1 - (t1 - t0) * (1 + math.cos(math.pi * progress)) / 2
        return self._ema_decay

    @torch.no_grad()
    def update_ema(self):
        tau = self._ema_decay
        ctx_params = dict(self.context_encoder.named_parameters())
        for name, tp in self.target_encoder.named_parameters():
            if name in ctx_params:
                tp.data.mul_(tau).add_(ctx_params[name].data, alpha=1.0 - tau)

    def _extract_visible_and_target(
        self,
        tokens: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract visible and target subsequences with position IDs.

        Returns padded tensors for batched processing.
        """
        B, L = tokens.shape
        device = tokens.device
        valid = tokens != self.pad_token_id  # (B, L)
        visible_mask = valid & ~target_mask  # (B, L)

        # Count max lengths for padding
        vis_counts = visible_mask.sum(dim=1)
        tgt_counts = target_mask.sum(dim=1)
        max_vis = vis_counts.max().item()
        max_tgt = tgt_counts.max().item()

        # Ensure minimum sizes
        max_vis = max(max_vis, 1)
        max_tgt = max(max_tgt, 1)

        vis_tokens = torch.full((B, max_vis), self.pad_token_id, device=device, dtype=torch.long)
        vis_positions = torch.zeros(B, max_vis, device=device, dtype=torch.long)
        tgt_positions = torch.zeros(B, max_tgt, device=device, dtype=torch.long)
        tgt_padding = torch.ones(B, max_tgt, device=device, dtype=torch.bool)  # True=pad

        for b in range(B):
            vis_idx = visible_mask[b].nonzero(as_tuple=True)[0]
            tgt_idx = target_mask[b].nonzero(as_tuple=True)[0]

            n_vis = len(vis_idx)
            n_tgt = len(tgt_idx)

            if n_vis > 0:
                vis_tokens[b, :n_vis] = tokens[b, vis_idx]
                vis_positions[b, :n_vis] = vis_idx

            if n_tgt > 0:
                tgt_positions[b, :n_tgt] = tgt_idx
                tgt_padding[b, :n_tgt] = False

        return vis_tokens, vis_positions, tgt_positions, tgt_padding, visible_mask

    def _apply_mlm_mask(
        self, vis_tokens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply 15% random masking within visible tokens for MLM anti-collapse.

        Returns:
            mlm_tokens: visible tokens with 15% replaced by [MASK]
            mlm_mask: (B, L_vis) True at MLM-masked positions
            mlm_labels: original token IDs at masked positions, -100 elsewhere
        """
        B, L = vis_tokens.shape
        device = vis_tokens.device

        valid = vis_tokens != self.pad_token_id
        mlm_mask = torch.zeros(B, L, dtype=torch.bool, device=device)
        mlm_labels = torch.full((B, L), -100, dtype=torch.long, device=device)

        # Vectorized: sample mask probabilities
        probs = torch.rand(B, L, device=device)
        mlm_mask = (probs < self.mlm_mask_ratio) & valid

        mlm_labels[mlm_mask] = vis_tokens[mlm_mask]
        mlm_tokens = vis_tokens.clone()
        mlm_tokens[mlm_mask] = self.mask_token_id

        return mlm_tokens, mlm_mask, mlm_labels

    def forward(
        self,
        tokens: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            tokens: (B, L) original token IDs
            target_mask: (B, L) True at target (JEPA-masked) positions

        Returns dict with all loss inputs.
        """
        B, L = tokens.shape

        # 1. Extract visible and target subsequences
        vis_tokens, vis_positions, tgt_positions, tgt_padding, visible_mask = \
            self._extract_visible_and_target(tokens, target_mask)

        # 2. Apply MLM masking within visible tokens (anti-collapse)
        mlm_tokens, mlm_mask, mlm_labels = self._apply_mlm_mask(vis_tokens)

        # 3. Context encoder: only visible tokens (with MLM masks applied)
        ctx_out = self.context_encoder(mlm_tokens, vis_positions)
        context_cls = ctx_out["cls"]         # (B, D)
        context_tokens = ctx_out["tokens"]   # (B, L_vis, D)

        # 4. MLM logits from visible token embeddings
        mlm_logits = self.mlm_head(context_tokens)  # (B, L_vis, vocab)

        # 5. Target encoder: full input (no grad)
        with torch.no_grad():
            tgt_out = self.target_encoder(tokens)
            target_tokens_emb = tgt_out["tokens"]  # (B, L, D)

        # 6. JEPA predictor: predict target embeddings from context
        jepa_predictions = self.predictor(
            context_emb=context_tokens,
            context_positions=vis_positions,
            target_positions=tgt_positions,
            target_padding_mask=tgt_padding,
        )  # (B, L_tgt, D)

        # 7. Gather actual target embeddings at target positions
        # target_tokens_emb is (B, L, D), tgt_positions is (B, L_tgt)
        tgt_pos_expanded = tgt_positions.unsqueeze(-1).expand(-1, -1, self.embed_dim)
        jepa_targets = torch.gather(target_tokens_emb, 1, tgt_pos_expanded)  # (B, L_tgt, D)

        return {
            "jepa_predictions": jepa_predictions,      # (B, L_tgt, D)
            "jepa_targets": jepa_targets.detach(),      # (B, L_tgt, D)
            "jepa_target_padding": tgt_padding,         # (B, L_tgt) True=pad
            "mlm_logits": mlm_logits,                   # (B, L_vis, vocab)
            "mlm_labels": mlm_labels,                   # (B, L_vis)
            "mlm_mask": mlm_mask,                       # (B, L_vis)
            "context_cls": context_cls,                  # (B, D)
        }

    @torch.no_grad()
    def encode(self, tokens: torch.Tensor, use_target: bool = True) -> torch.Tensor:
        """Extract CLS embeddings for evaluation."""
        if use_target:
            return self.target_encoder(tokens)["cls"]
        else:
            B, L = tokens.shape
            valid = tokens != self.pad_token_id
            positions = torch.arange(L, device=tokens.device).unsqueeze(0).expand(B, -1)
            return self.context_encoder(tokens, positions)["cls"]


# =============================================================================
# 10. Loss Computation
# =============================================================================

def compute_losses(
    model_out: Dict[str, torch.Tensor],
    model: BJEPAv6,
    tokens: torch.Tensor,
    gc_target: torch.Tensor,
    gc_lambda: float,
    args,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute all v6 losses. JEPA is PRIMARY, MLM is anti-collapse."""
    metrics = {}

    # -- 1. JEPA Loss (PRIMARY) --
    pred = model_out["jepa_predictions"]     # (B, L_tgt, D)
    target = model_out["jepa_targets"]       # (B, L_tgt, D)
    tgt_pad = model_out["jepa_target_padding"]  # (B, L_tgt)

    # Smooth L1 at non-padded target positions
    valid_tgt = ~tgt_pad  # True = real target
    if valid_tgt.any():
        pred_valid = pred[valid_tgt]      # (N, D)
        target_valid = target[valid_tgt]  # (N, D)
        jepa_loss = F.smooth_l1_loss(pred_valid, target_valid)

        with torch.no_grad():
            jepa_cos = F.cosine_similarity(pred_valid, target_valid, dim=-1).mean().item()
            jepa_l2 = (pred_valid - target_valid).pow(2).mean().item()
    else:
        jepa_loss = torch.tensor(0.0, device=tokens.device)
        jepa_cos = 0.0
        jepa_l2 = 0.0

    metrics["jepa_loss"] = jepa_loss.item()
    metrics["jepa_cos_sim"] = jepa_cos
    metrics["jepa_l2"] = jepa_l2
    metrics["n_target_tokens"] = valid_tgt.sum().item()

    # -- 2. MLM Loss (ANTI-COLLAPSE GUARD) --
    logits = model_out["mlm_logits"]  # (B, L_vis, V)
    labels = model_out["mlm_labels"]  # (B, L_vis)
    mlm_mask = model_out["mlm_mask"]  # (B, L_vis)

    B_mlm, L_mlm, V = logits.shape
    logits_flat = logits.reshape(-1, V)
    labels_flat = labels.reshape(-1)
    mask_flat = mlm_mask.reshape(-1)

    mlm_loss_all = F.cross_entropy(logits_flat, labels_flat, reduction='none', ignore_index=-100)
    n_mlm_masked = mask_flat.sum().clamp(min=1)
    mlm_loss = (mlm_loss_all * mask_flat.float()).sum() / n_mlm_masked

    with torch.no_grad():
        preds = logits_flat.argmax(dim=-1)
        mlm_acc = ((preds == labels_flat) & mask_flat).sum().float() / n_mlm_masked

    metrics["mlm_loss"] = mlm_loss.item()
    metrics["mlm_acc"] = mlm_acc.item()
    metrics["n_mlm_masked"] = n_mlm_masked.item()

    # -- 3. SIGReg on context CLS --
    sigreg_loss, sigreg_met = model.sigreg(model_out["context_cls"])
    metrics.update({f"sigreg_{k}": v for k, v in sigreg_met.items()})

    # -- 4. GC Adversary --
    gc_adv_loss = torch.tensor(0.0, device=tokens.device)
    if args.gc_adv_weight > 0 and gc_lambda > 0:
        gc_pred = model.gc_adversary(model_out["context_cls"], lam=gc_lambda)
        gc_adv_loss = F.mse_loss(gc_pred, gc_target)
    metrics["gc_adv_loss"] = gc_adv_loss.item()

    # -- Total: JEPA-dominant --
    total = (
        args.jepa_weight * jepa_loss
        + args.mlm_weight * mlm_loss
        + args.sigreg_weight * sigreg_loss
        + args.gc_adv_weight * gc_adv_loss
    )
    metrics["total_loss"] = total.item()

    return total, metrics


# =============================================================================
# 11. GC Content Computation (vectorized)
# =============================================================================

def compute_gc_content(tokens: torch.Tensor, pad_id: int, gc_ids: Set[int]) -> torch.Tensor:
    """Per-sequence GC fraction. Vectorized over gc_ids."""
    valid = tokens != pad_id
    lengths = valid.sum(dim=1).float().clamp(min=1)
    gc_ids_t = torch.tensor(list(gc_ids), device=tokens.device, dtype=torch.long)
    gc_mask = (tokens.unsqueeze(-1) == gc_ids_t.unsqueeze(0).unsqueeze(0)).any(dim=-1)
    return (gc_mask & valid).sum(dim=1).float() / lengths


# =============================================================================
# 12. RankMe + GC Correlation
# =============================================================================

@torch.no_grad()
def compute_rankme(embeddings: torch.Tensor) -> float:
    X = embeddings.float()
    X = X - X.mean(dim=0, keepdim=True)
    try:
        s = torch.linalg.svdvals(X)
    except Exception:
        _, s, _ = torch.svd(X)
    s = s[s > 1e-7]
    p = s / s.sum()
    return math.exp(-(p * p.log()).sum().item())


def gc_correlation(tokens, embeddings, pad_id, gc_ids):
    gc = compute_gc_content(tokens, pad_id, gc_ids).cpu().numpy()
    emb = embeddings.float().cpu().numpy()
    try:
        from numpy.linalg import svd
        _, _, Vt = svd(emb - emb.mean(axis=0), full_matrices=False)
        pc1 = emb @ Vt[0]
        r = float(np.corrcoef(gc, pc1)[0, 1])
        return abs(r)
    except Exception:
        return 0.0


# =============================================================================
# 13. Visualization
# =============================================================================

@torch.no_grad()
def generate_viz(emb_np, gc_np, save_path, epoch):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import umap
        from sklearn.manifold import TSNE
    except ImportError:
        print("  Viz skipped — install umap-learn, sklearn, matplotlib")
        return

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    try:
        reducer = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.3,
                            metric='cosine', random_state=42)
        coords = reducer.fit_transform(emb_np)
        sc = axes[0].scatter(coords[:, 0], coords[:, 1], c=gc_np,
                             cmap='RdYlBu_r', s=3, alpha=0.6)
        plt.colorbar(sc, ax=axes[0], label='GC content')
        axes[0].set_title(f'UMAP — Epoch {epoch} (GC%)')
    except Exception as e:
        axes[0].text(0.5, 0.5, f'Failed: {e}', transform=axes[0].transAxes, ha='center')

    try:
        n = min(3000, len(emb_np))
        tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42)
        coords = tsne.fit_transform(emb_np[:n])
        sc = axes[1].scatter(coords[:, 0], coords[:, 1], c=gc_np[:n],
                             cmap='RdYlBu_r', s=3, alpha=0.6)
        plt.colorbar(sc, ax=axes[1], label='GC content')
        axes[1].set_title(f't-SNE — Epoch {epoch} (GC%)')
    except Exception as e:
        axes[1].text(0.5, 0.5, f'Failed: {e}', transform=axes[1].transAxes, ha='center')

    for ax in axes:
        ax.set_xlabel('Dim 1')
        ax.set_ylabel('Dim 2')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved viz: {save_path}")


# =============================================================================
# 14. Dataset
# =============================================================================

class BPEPretrainDataset(Dataset):
    def __init__(self, csv_path: str, tokenizer_path: str, max_len: int = 512):
        import pandas as pd
        import json
        self.df = pd.read_csv(csv_path)
        self.max_len = max_len

        with open(tokenizer_path) as f:
            tok_data = json.load(f)
        self.vocab = tok_data.get('model', tok_data).get('vocab', tok_data.get('vocab', {}))
        self.pad_id = 0
        self.mask_id = self.vocab.get("[MASK]", 1)

        try:
            from tokenizers import Tokenizer
            self.tokenizer = Tokenizer.from_file(tokenizer_path)
            self._encode = lambda seq: self.tokenizer.encode(seq).ids[:self.max_len]
        except ImportError:
            self._encode = lambda seq: [self.vocab.get(c, 1) for c in seq.upper()][:self.max_len]

        # GC token IDs (vectorized later)
        self.gc_token_ids = set()
        for key, val in self.vocab.items():
            k = key.upper().replace('\u0120', '').replace('\u2581', '')
            if k and all(c in 'GC' for c in k):
                self.gc_token_ids.add(val)
        if not self.gc_token_ids:
            self.gc_token_ids = {2, 3}

        seq_col = 'sequence' if 'sequence' in self.df.columns else self.df.columns[0]
        self.sequences = self.df[seq_col].values
        print(f"  Dataset: {len(self):,} sequences, max_len={max_len}")
        print(f"  GC token IDs: {len(self.gc_token_ids)} tokens")
        print(f"  Mask token ID: {self.mask_id}")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        ids = self._encode(str(self.sequences[idx]))
        if len(ids) < self.max_len:
            ids = ids + [self.pad_id] * (self.max_len - len(ids))
        return torch.tensor(ids, dtype=torch.long)


# =============================================================================
# 15. Training Utilities
# =============================================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cosine_lr(optimizer, step, total, warmup, peak, floor=1e-6):
    if step < warmup:
        lr = peak * step / max(warmup, 1)
    else:
        p = (step - warmup) / max(total - warmup, 1)
        lr = floor + 0.5 * (peak - floor) * (1 + math.cos(math.pi * p))
    for pg in optimizer.param_groups:
        if "lr_scale" in pg:
            pg["lr"] = lr * pg["lr_scale"]
        else:
            pg["lr"] = lr
    return lr


# =============================================================================
# 16. Evaluation
# =============================================================================

@torch.no_grad()
def evaluate(model, dl, device, gc_ids, pad_id, max_batches=30):
    model.eval()
    all_emb, all_tok = [], []
    for i, batch in enumerate(dl):
        if i >= max_batches:
            break
        tokens = batch.to(device)
        emb = model.encode(tokens, use_target=True)
        all_emb.append(emb.cpu())
        all_tok.append(tokens.cpu())

    embs = torch.cat(all_emb, 0)
    toks = torch.cat(all_tok, 0)
    rankme = compute_rankme(embs)
    gc_r = gc_correlation(toks, embs, pad_id, gc_ids)
    std = embs.std(dim=0).mean().item()
    norm = embs.norm(dim=1).mean().item()

    return {
        "rankme": rankme, "gc_abs_r": gc_r,
        "std": std, "norm": norm,
        "embeddings": embs, "tokens": toks,
    }


# =============================================================================
# 17. Main Training Loop
# =============================================================================

def train(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'=' * 70}")
    print(f"  B-JEPA v6.0 — True JEPA for Bacterial DNA")
    print(f"{'=' * 70}")
    print(f"  Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    # Dataset
    dataset = BPEPretrainDataset(args.data_path, args.tokenizer_path, args.max_seq_len)
    pad_id = dataset.pad_id
    mask_id = dataset.mask_id
    gc_ids = dataset.gc_token_ids
    vocab_size = len(dataset.vocab)

    dl_kwargs = dict(
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    if args.num_workers > 0:
        dl_kwargs["prefetch_factor"] = 2
    dl = DataLoader(dataset, **dl_kwargs)

    # Model
    model = BJEPAv6(
        vocab_size=vocab_size,
        embed_dim=args.embed_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        ff_dim=args.ff_dim,
        max_seq_len=args.max_seq_len,
        predictor_dim=args.predictor_dim,
        predictor_depth=args.predictor_depth,
        predictor_heads=args.predictor_heads,
        pad_token_id=pad_id,
        mask_token_id=mask_id,
        ema_start=args.ema_start,
        ema_end=args.ema_end,
        mlm_mask_ratio=args.mlm_mask_ratio,
        var_gamma=args.var_gamma,
        qk_norm=True,
        bias=False,
    ).to(device)

    n_ctx = sum(p.numel() for p in model.context_encoder.parameters())
    n_pred = sum(p.numel() for p in model.predictor.parameters())
    n_mlm = sum(p.numel() for p in model.mlm_head.parameters())
    n_total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  Context Encoder: {n_ctx/1e6:.2f}M")
    print(f"  JEPA Predictor:  {n_pred/1e6:.2f}M")
    print(f"  MLM Head:        {n_mlm/1e6:.2f}M")
    print(f"  Total trainable: {n_total/1e6:.2f}M")
    print(f"  Architecture:    {args.num_layers}L x {args.embed_dim}D x {args.num_heads}H")
    print(f"                   RoPE + RMSNorm + SwiGLU + QK-Norm")
    print(f"  Predictor:       {args.predictor_depth}L x {args.predictor_dim}D (cross-attention)")

    # Optimizer — separate LR for adversary
    opt = torch.optim.AdamW([
        {"params": model.context_encoder.parameters()},
        {"params": model.predictor.parameters()},
        {"params": model.mlm_head.parameters()},
        {"params": model.gc_adversary.parameters(), "lr_scale": 5.0},
    ], lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95))

    steps_per_epoch = len(dl)
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = args.warmup_epochs * steps_per_epoch

    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler('cuda', enabled=(use_amp and amp_dtype == torch.float16))

    ckpt_dir = Path(args.output_dir) / "checkpoints" / "v6.0"
    viz_dir = ckpt_dir / "viz"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  Training: {args.epochs} epochs, warmup={args.warmup_epochs}")
    print(f"  LR: {args.lr:.1e} -> {args.min_lr:.1e} (cosine)")
    print(f"  JEPA masking: {args.jepa_mask_start:.0%} -> {args.jepa_mask_end:.0%} (curriculum)")
    print(f"  MLM masking:  {args.mlm_mask_ratio:.0%} (within visible, anti-collapse)")
    print(f"  Block config: {args.num_blocks} blocks, min_len {args.min_block_start}->{args.min_block_end}")
    print(f"  Loss weights: JEPA={args.jepa_weight} MLM={args.mlm_weight} "
          f"SIGReg={args.sigreg_weight} GC_adv={args.gc_adv_weight}")
    print(f"  Steps/epoch: {steps_per_epoch}  Total: {total_steps}")
    print(f"{'=' * 70}\n")

    use_wandb = HAS_WANDB and not args.no_wandb
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name or f"bjepa-v6.0-true-jepa-{args.num_layers}L{args.embed_dim}D",
            config=vars(args),
        )

    global_step = 0
    best_loss = float('inf')
    t_start = time.time()

    for epoch in range(args.epochs):
        epoch_t = time.time()
        model.train()
        progress = epoch / max(args.epochs - 1, 1)
        ema_tau = model.set_ema_decay(progress)

        # Curriculum: mask ratio and block length
        jepa_mr = curriculum_schedule(epoch, args.epochs, args.jepa_mask_start, args.jepa_mask_end)
        min_blen = int(curriculum_schedule(epoch, args.epochs, args.min_block_start, args.min_block_end))
        min_blen = max(1, min_blen)
        gc_lam = GCAdversary.ganin_lambda(epoch, args.epochs) if args.gc_adv_weight > 0 else 0.0

        epoch_metrics = {}
        opt.zero_grad(set_to_none=True)

        pbar = tqdm(dl, desc=f"Epoch {epoch+1}/{args.epochs}", leave=True, ncols=160)

        for step, tokens in enumerate(pbar, start=1):
            global_step += 1
            lr = cosine_lr(opt, global_step, total_steps, warmup_steps, args.lr, args.min_lr)
            tokens = tokens.to(device, non_blocking=True)
            valid = tokens != pad_id

            # Multi-block masking for JEPA
            target_mask = multi_block_mask(
                tokens.shape[1], jepa_mr, args.num_blocks,
                min_blen, valid, device,
            )

            amp_ctx = torch.autocast("cuda", dtype=amp_dtype) if use_amp else nullcontext()
            with amp_ctx:
                out = model(tokens, target_mask)

                gc_target = compute_gc_content(tokens, pad_id, gc_ids)
                total, met = compute_losses(out, model, tokens, gc_target, gc_lam, args)

            if scaler.is_enabled():
                scaler.scale(total).backward()
            else:
                total.backward()

            if step % args.grad_accum == 0 or step == steps_per_epoch:
                if scaler.is_enabled():
                    scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], args.grad_clip
                )
                if scaler.is_enabled():
                    scaler.step(opt)
                    scaler.update()
                else:
                    opt.step()
                opt.zero_grad(set_to_none=True)
                model.update_ema()

            for k, v in met.items():
                epoch_metrics.setdefault(k, []).append(v)

            if step % args.log_every == 0:
                pbar.set_postfix({
                    "loss": f"{met['total_loss']:.4f}",
                    "jepa": f"{met['jepa_loss']:.3f}",
                    "jcos": f"{met['jepa_cos_sim']:.3f}",
                    "mlm": f"{met['mlm_loss']:.3f}",
                    "acc": f"{met['mlm_acc']:.3f}",
                    "sig": f"{met.get('sigreg_sigreg', 0):.4f}",
                    "vf": f"{met.get('sigreg_var_floor', 0):.3f}",
                    "mr": f"{jepa_mr:.2f}",
                    "lr": f"{lr:.1e}",
                })

                if use_wandb:
                    wandb.log(
                        {f"train/{k}": v for k, v in met.items()}
                        | {"schedule/lr": lr, "schedule/jepa_mask_ratio": jepa_mr,
                           "schedule/min_block_len": min_blen},
                        step=global_step,
                    )

        # -- Epoch eval --
        ep_time = time.time() - epoch_t
        avg = {k: sum(v) / len(v) for k, v in epoch_metrics.items()}
        eval_met = evaluate(model, dl, device, gc_ids, pad_id)

        elapsed = time.time() - t_start
        eta = (elapsed / (epoch + 1)) * (args.epochs - epoch - 1)

        print(f"\n  Epoch {epoch+1}/{args.epochs} ({ep_time:.0f}s, ETA {int(eta//3600)}:{int(eta%3600//60):02d})")
        print(f"  JEPA:   loss={avg['jepa_loss']:.4f}  cos_sim={avg['jepa_cos_sim']:.3f}  "
              f"l2={avg['jepa_l2']:.4f}  n_tgt={avg['n_target_tokens']:.0f}")
        print(f"  MLM:    loss={avg['mlm_loss']:.4f}  acc={avg['mlm_acc']:.3f}")
        print(f"  SIGReg: {avg.get('sigreg_sigreg', 0):.4f}  var_floor={avg.get('sigreg_var_floor', 0):.4f}  std={avg.get('sigreg_std_mean', 0):.3f}")
        print(f"  GC_adv: {avg['gc_adv_loss']:.5f}")
        print(f"  RankMe: {eval_met['rankme']:.1f}/{args.embed_dim}  "
              f"GC|r|={eval_met['gc_abs_r']:.3f}  std={eval_met['std']:.3f}  "
              f"norm={eval_met['norm']:.1f}")
        print(f"  Mask: {jepa_mr:.3f}  BlockLen>={min_blen}  LR={lr:.2e}  EMA={ema_tau:.4f}")

        if eval_met['rankme'] < 50:
            print(f"  WARNING: RankMe={eval_met['rankme']:.1f} < 50 — possible collapse!")
        elif eval_met['rankme'] < 200:
            print(f"  WATCH:   RankMe={eval_met['rankme']:.1f} — monitor closely")
        else:
            print(f"  HEALTHY: RankMe={eval_met['rankme']:.1f}")

        # Viz
        try:
            emb_np = eval_met['embeddings'].float().numpy()
            gc_np = compute_gc_content(eval_met['tokens'], pad_id, gc_ids).numpy()
            n_viz = min(3000, len(emb_np))
            idx = np.random.choice(len(emb_np), n_viz, replace=False)
            viz_path = str(viz_dir / f"embeddings_epoch{epoch+1:03d}.png")
            generate_viz(emb_np[idx], gc_np[idx], viz_path, epoch + 1)
        except Exception as e:
            print(f"  Viz failed: {e}")

        if use_wandb:
            log = {f"epoch/{k}": v for k, v in avg.items()}
            log.update({
                "health/rankme": eval_met["rankme"],
                "health/gc_abs_r": eval_met["gc_abs_r"],
                "health/std": eval_met["std"],
                "health/norm": eval_met["norm"],
                "schedule/ema_tau": ema_tau,
                "epoch": epoch + 1,
            })
            vp = str(viz_dir / f"embeddings_epoch{epoch+1:03d}.png")
            if os.path.exists(vp):
                log["viz/embeddings"] = wandb.Image(vp)
            wandb.log(log, step=global_step)

        if avg['total_loss'] < best_loss:
            best_loss = avg['total_loss']

        # Checkpoint
        if (epoch + 1) % args.save_every == 0 or (epoch + 1) == args.epochs:
            path = ckpt_dir / f"epoch{epoch+1:04d}.pt"
            torch.save({
                "epoch": epoch + 1,
                "global_step": global_step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "scaler_state_dict": scaler.state_dict() if scaler.is_enabled() else None,
                "metrics": {k: v for k, v in {**avg, **eval_met}.items()
                            if not isinstance(v, torch.Tensor)},
                "config": vars(args),
                "version": "v6.0-true-jepa",
            }, path)
            print(f"  Saved: {path}")

    total_time = time.time() - t_start
    h, m = int(total_time // 3600), int(total_time % 3600 // 60)
    print(f"\n{'=' * 70}")
    print(f"  B-JEPA v6.0 complete in {h}h {m}m")
    print(f"  Best loss: {best_loss:.4f}  Final RankMe: {eval_met['rankme']:.1f}")
    print(f"{'=' * 70}")
    if use_wandb:
        wandb.finish()


# =============================================================================
# 18. CLI
# =============================================================================

def build_parser():
    p = argparse.ArgumentParser(description="B-JEPA v6.0 — True JEPA for Bacterial DNA")

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
    g.add_argument("--grad-accum", type=int, default=1)

    g = p.add_argument_group("Encoder (RoPE + RMSNorm + SwiGLU + QK-Norm)")
    g.add_argument("--embed-dim", type=int, default=576)
    g.add_argument("--num-layers", type=int, default=12)
    g.add_argument("--num-heads", type=int, default=9)
    g.add_argument("--ff-dim", type=int, default=2304)
    g.add_argument("--max-seq-len", type=int, default=512)

    g = p.add_argument_group("JEPA Predictor (cross-attention)")
    g.add_argument("--predictor-dim", type=int, default=384)
    g.add_argument("--predictor-depth", type=int, default=6)
    g.add_argument("--predictor-heads", type=int, default=6)
    g.add_argument("--ema-start", type=float, default=0.996)
    g.add_argument("--ema-end", type=float, default=1.0)

    g = p.add_argument_group("JEPA Masking (multi-block, curriculum)")
    g.add_argument("--jepa-mask-start", type=float, default=0.50,
                    help="JEPA target mask ratio at epoch 0")
    g.add_argument("--jepa-mask-end", type=float, default=0.70,
                    help="JEPA target mask ratio at final epoch")
    g.add_argument("--num-blocks", type=int, default=4,
                    help="Number of target blocks for multi-block masking")
    g.add_argument("--min-block-start", type=int, default=10,
                    help="Minimum block length at epoch 0")
    g.add_argument("--min-block-end", type=int, default=30,
                    help="Minimum block length at final epoch")

    g = p.add_argument_group("MLM Anti-Collapse")
    g.add_argument("--mlm-mask-ratio", type=float, default=0.15,
                    help="Random mask ratio within visible tokens")

    g = p.add_argument_group("Loss Weights")
    g.add_argument("--jepa-weight", type=float, default=5.0,
                    help="JEPA loss weight (PRIMARY)")
    g.add_argument("--mlm-weight", type=float, default=0.5,
                    help="MLM loss weight (anti-collapse guard)")
    g.add_argument("--sigreg-weight", type=float, default=10.0,
                    help="SIGReg regularization weight")
    g.add_argument("--var-gamma", type=float, default=1.0,
                    help="Variance floor threshold for SIGReg (hinge on projection std)")
    g.add_argument("--gc-adv-weight", type=float, default=1.0,
                    help="GC adversary weight")

    g = p.add_argument_group("Logging")
    g.add_argument("--save-every", type=int, default=5)
    g.add_argument("--log-every", type=int, default=50)
    g.add_argument("--num-workers", type=int, default=4)
    g.add_argument("--no-wandb", action="store_true")
    g.add_argument("--wandb-project", default="bdna-jepa")
    g.add_argument("--wandb-run-name", default=None)

    return p


if __name__ == "__main__":
    train(build_parser().parse_args())
