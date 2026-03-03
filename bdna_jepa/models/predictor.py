"""JEPA predictors: narrow bottleneck for latent-space prediction.

Predictor          — Asymmetric CLS->CLS predictor (I-JEPA style)
FragmentPredictor  — Cross-fragment genome-level predictor (v4.0+)
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from bdna_jepa.config import PredictorConfig, FragmentConfig
from bdna_jepa.models.encoder import get_norm, SwiGLU, GELUMLP


class PredictorBlock(nn.Module):
    """Lightweight transformer block for the predictor."""

    def __init__(self, dim: int, num_heads: int, ff_dim: int, ff_activation: str,
                 norm_type: str, dropout: float, bias: bool):
        super().__init__()
        self.norm1 = get_norm(norm_type, dim)
        self.attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True, bias=bias,
        )
        self.norm2 = get_norm(norm_type, dim)
        if ff_activation == "swiglu":
            self.mlp = SwiGLU(dim, ff_dim, bias=bias)
        else:
            self.mlp = GELUMLP(dim, ff_dim, bias=bias, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + self.dropout(h)
        x = x + self.dropout(self.mlp(self.norm2(x)))
        return x


class Predictor(nn.Module):
    """Asymmetric CLS -> CLS predictor with bottleneck.

    project_in(D -> D_pred) -> transformer layers -> project_out(D_pred -> D)
    Bottleneck ratio 0.33x forces encoder to build richer representations.
    """

    def __init__(self, encoder_dim: int, config: PredictorConfig):
        super().__init__()
        self.project_in = nn.Linear(encoder_dim, config.dim, bias=config.bias)

        self.blocks = nn.ModuleList([
            PredictorBlock(
                dim=config.dim,
                num_heads=config.num_heads,
                ff_dim=config.ff_dim,
                ff_activation=config.ff_activation,
                norm_type=config.norm_type,
                dropout=config.dropout,
                bias=config.bias,
            )
            for _ in range(config.depth)
        ])

        self.norm = get_norm(config.norm_type, config.dim)
        self.project_out = nn.Linear(config.dim, encoder_dim, bias=config.bias)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, cls_embedding: torch.Tensor) -> torch.Tensor:
        """Predict target CLS from context CLS.

        Args:
            cls_embedding: (B, D) context encoder CLS output

        Returns:
            (B, D) predicted target CLS embedding
        """
        x = self.project_in(cls_embedding).unsqueeze(1)  # (B, 1, D_pred)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        x = self.project_out(x).squeeze(1)  # (B, D)
        return x


class FragmentPredictor(nn.Module):
    """Cross-fragment genome-level predictor.

    Given K-1 context fragment CLS embeddings from the same genome,
    predicts the held-out fragment CLS via cross-attention.
    """

    def __init__(self, encoder_dim: int, config: FragmentConfig):
        super().__init__()
        dim = config.predictor_dim

        self.project_in = nn.Linear(encoder_dim, dim)
        self.query_token = nn.Parameter(torch.randn(1, 1, dim) * 0.02)

        self.cross_attn_layers = nn.ModuleList()
        self.ff_layers = nn.ModuleList()
        self.norms1 = nn.ModuleList()
        self.norms2 = nn.ModuleList()

        for _ in range(config.predictor_depth):
            self.cross_attn_layers.append(
                nn.MultiheadAttention(dim, config.predictor_heads, batch_first=True)
            )
            self.ff_layers.append(
                nn.Sequential(
                    nn.Linear(dim, dim * 4),
                    nn.GELU(),
                    nn.Linear(dim * 4, dim),
                )
            )
            self.norms1.append(nn.LayerNorm(dim))
            self.norms2.append(nn.LayerNorm(dim))

        self.project_out = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, encoder_dim),
        )

    def forward(
        self,
        context_embeddings: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Predict held-out fragment from context fragments.

        Args:
            context_embeddings: (B, K-1, D) CLS embeddings of context fragments
            context_mask: (B, K-1) bool mask, True = valid

        Returns:
            (B, D) predicted target fragment CLS
        """
        B = context_embeddings.size(0)
        kv = self.project_in(context_embeddings)  # (B, K-1, dim)
        query = self.query_token.expand(B, -1, -1)  # (B, 1, dim)

        key_padding_mask = None
        if context_mask is not None:
            key_padding_mask = ~context_mask  # MHA expects True = ignore

        for ca, ff, n1, n2 in zip(
            self.cross_attn_layers, self.ff_layers, self.norms1, self.norms2
        ):
            h = n1(query)
            h, _ = ca(h, kv, kv, key_padding_mask=key_padding_mask, need_weights=False)
            query = query + h
            query = query + ff(n2(query))

        return self.project_out(query.squeeze(1))  # (B, D)
