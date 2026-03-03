"""Transformer encoder backbone for B-JEPA.

Supports v3.1 (char-level, learned pos, GELU, LayerNorm) and
v4.0 (BPE, RoPE, SwiGLU, RMSNorm, QK-Norm).

Architecture references:
    - Pre-norm: GPT-2, Llama, PaLM
    - RoPE: Su et al., 2021; used by Llama, Mistral
    - SwiGLU: Shazeer, 2020; Llama, PaLM
    - RMSNorm: Zhang & Sennrich, 2019; Llama
    - QK-Norm: Dehghani et al., 2023; ViT-22B
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from bdna_jepa.config import EncoderConfig


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (Zhang & Sennrich, 2019)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).type_as(x) * self.weight


def get_norm(norm_type: str, dim: int) -> nn.Module:
    if norm_type == "rmsnorm":
        return RMSNorm(dim)
    return nn.LayerNorm(dim)


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

    def forward(self, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self.cos_cached.size(0):
            self._build_cache(seq_len)
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    # q, k: (B, H, L, D)  cos, sin: (L, D)
    cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, L, D)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_rot = q * cos + rotate_half(q) * sin
    k_rot = k * cos + rotate_half(k) * sin
    return q_rot, k_rot


class SwiGLU(nn.Module):
    """SwiGLU feedforward (Shazeer, 2020). Used by Llama, PaLM."""

    def __init__(self, dim: int, ff_dim: int, bias: bool = False):
        super().__init__()
        # SwiGLU uses 2/3 of ff_dim for the actual hidden size
        hidden = int(2 * ff_dim / 3)
        # Round to nearest multiple of 8 for efficiency
        hidden = ((hidden + 7) // 8) * 8
        self.w1 = nn.Linear(dim, hidden, bias=bias)
        self.w2 = nn.Linear(hidden, dim, bias=bias)
        self.w3 = nn.Linear(dim, hidden, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class GELUMLP(nn.Module):
    """Standard GELU feedforward."""

    def __init__(self, dim: int, ff_dim: int, bias: bool = True, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, ff_dim, bias=bias)
        self.fc2 = nn.Linear(ff_dim, dim, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.dropout(F.gelu(self.fc1(x))))


class MultiHeadAttention(nn.Module):
    """Multi-head attention with optional RoPE and QK-Norm."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        qk_norm: bool = False,
        bias: bool = False,
        attention_dropout: float = 0.0,
        norm_type: str = "rmsnorm",
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, 3 * dim, bias=bias)
        self.out_proj = nn.Linear(dim, dim, bias=bias)
        self.attn_dropout = nn.Dropout(attention_dropout)

        self.qk_norm = qk_norm
        if qk_norm:
            self.q_norm = get_norm(norm_type, self.head_dim)
            self.k_norm = get_norm(norm_type, self.head_dim)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, L, D = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(2)  # each: (B, L, H, D)
        q = q.transpose(1, 2)  # (B, H, L, D)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        if rope_cos is not None and rope_sin is not None:
            q, k = apply_rotary_pos_emb(q, k, rope_cos, rope_sin)

        # Scaled dot-product attention (uses Flash Attention when available)
        attn_mask = None
        if attention_mask is not None:
            # attention_mask: (B, L) bool, True = attend, False = ignore
            # Expand for (B, H, L_q, L_kv) format
            attn_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, L)
            attn_mask = attn_mask.expand(-1, -1, L, -1)
            attn_mask = attn_mask.to(dtype=q.dtype)
            attn_mask = attn_mask.masked_fill(attn_mask == 0, float("-inf"))
            attn_mask = attn_mask.masked_fill(attn_mask == 1, 0.0)

        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=self.attn_dropout.p if self.training else 0.0
        )

        out = out.transpose(1, 2).reshape(B, L, D)
        return self.out_proj(out)


class TransformerBlock(nn.Module):
    """Pre-norm transformer block."""

    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.norm1 = get_norm(config.norm_type, config.embed_dim)
        self.attn = MultiHeadAttention(
            dim=config.embed_dim,
            num_heads=config.num_heads,
            qk_norm=config.qk_norm,
            bias=config.bias,
            attention_dropout=config.attention_dropout,
            norm_type=config.norm_type,
        )
        self.norm2 = get_norm(config.norm_type, config.embed_dim)

        if config.ff_activation == "swiglu":
            self.mlp = SwiGLU(config.embed_dim, config.ff_dim, bias=config.bias)
        else:
            self.mlp = GELUMLP(config.embed_dim, config.ff_dim, bias=config.bias, dropout=config.dropout)

        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = x + self.dropout(self.attn(self.norm1(x), attention_mask, rope_cos, rope_sin))
        x = x + self.dropout(self.mlp(self.norm2(x)))
        return x


class TransformerEncoder(nn.Module):
    """Full encoder: token_emb + CLS + pos_emb + layers + final_norm.

    Outputs per-token embeddings and a pooled [CLS] representation.
    """

    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.config = config

        self.token_emb = nn.Embedding(config.vocab_size, config.embed_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.embed_dim) * 0.02)

        # Positional encoding
        self.use_rope = config.pos_encoding == "rotary"
        if self.use_rope:
            self.rope = RotaryEmbedding(config.embed_dim // config.num_heads, config.max_seq_len + 1)
        else:
            self.pos_emb = nn.Embedding(config.max_seq_len + 1, config.embed_dim)  # +1 for CLS

        self.embed_dropout = nn.Dropout(config.embed_dropout)

        self.layers = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.num_layers)
        ])
        self.final_norm = get_norm(config.norm_type, config.embed_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.token_emb.weight, std=0.02)
        nn.init.normal_(self.cls_token, std=0.02)
        if not self.use_rope:
            nn.init.normal_(self.pos_emb.weight, std=0.02)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def get_num_params(self, non_embedding: bool = True) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.token_emb.weight.numel()
            if not self.use_rope and hasattr(self, "pos_emb"):
                n -= self.pos_emb.weight.numel()
        return n

    def forward(
        self,
        tokens: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_all_tokens: bool = False,
    ) -> dict[str, torch.Tensor]:
        B, L = tokens.shape
        x = self.token_emb(tokens)  # (B, L, D)

        # Prepend CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, D)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, L+1, D)

        # Positional encoding
        rope_cos, rope_sin = None, None
        if self.use_rope:
            rope_cos, rope_sin = self.rope(L + 1)
        else:
            positions = torch.arange(L + 1, device=tokens.device)
            x = x + self.pos_emb(positions)

        x = self.embed_dropout(x)

        # Extend attention mask for CLS token
        if attention_mask is not None:
            cls_mask = torch.ones(B, 1, device=tokens.device, dtype=attention_mask.dtype)
            attention_mask = torch.cat([cls_mask, attention_mask], dim=1)

        for layer in self.layers:
            x = layer(x, attention_mask, rope_cos, rope_sin)

        x = self.final_norm(x)

        cls_out = x[:, 0]  # (B, D)
        result = {"cls": cls_out}

        if return_all_tokens:
            result["tokens"] = x  # (B, L+1, D)

        return result

    def encode(self, tokens: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Convenience: returns CLS embedding only."""
        return self.forward(tokens, attention_mask)["cls"]
