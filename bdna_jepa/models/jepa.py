"""Full B-JEPA model: encoder + stop-grad target + predictor + MLM head.

Default: stop-gradient target (no EMA). EMA available as fallback via config.

Architecture:
    context_encoder  -> processes masked input (trainable)
    target_encoder   -> stop-grad copy OR EMA copy, processes full input (frozen)
    predictor        -> narrow bottleneck CLS->CLS prediction
    mlm_head         -> token-level prediction head
    fragment_pred    -> optional cross-fragment JEPA (v4.0+)
"""
from __future__ import annotations

import copy
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from bdna_jepa.config import BJEPAConfig
from bdna_jepa.models.encoder import TransformerEncoder, get_norm
from bdna_jepa.models.predictor import Predictor, FragmentPredictor


class MLMHead(nn.Module):
    """Masked Language Model prediction head.

    LayerNorm -> Linear -> GELU -> Linear(-> vocab_size)
    """

    def __init__(self, embed_dim: int, vocab_size: int, norm_type: str = "rmsnorm"):
        super().__init__()
        self.norm = get_norm(norm_type, embed_dim)
        self.dense = nn.Linear(embed_dim, embed_dim)
        self.activation = nn.GELU()
        self.output = nn.Linear(embed_dim, vocab_size)

        nn.init.normal_(self.dense.weight, std=0.02)
        nn.init.normal_(self.output.weight, std=0.02)
        if self.dense.bias is not None:
            nn.init.zeros_(self.dense.bias)
        if self.output.bias is not None:
            nn.init.zeros_(self.output.bias)

    def forward(self, token_embeddings: torch.Tensor) -> torch.Tensor:
        """(B, L, D) -> (B, L, vocab_size)"""
        x = self.norm(token_embeddings)
        x = self.activation(self.dense(x))
        return self.output(x)


class BJEPA(nn.Module):
    """B-JEPA: Bacterial Joint-Embedding Predictive Architecture.

    Components:
        context_encoder  — processes masked input (trainable)
        target_encoder   — stop-grad/EMA copy of context (frozen)
        predictor        — bottleneck CLS prediction
        mlm_head         — token-level MLM
        fragment_pred    — optional fragment-level JEPA
    """

    def __init__(self, config: BJEPAConfig):
        super().__init__()
        self.config = config

        # Context encoder (trainable)
        self.context_encoder = TransformerEncoder(config.encoder)

        # Target encoder (frozen — updated via stop-grad or EMA)
        self.target_encoder = copy.deepcopy(self.context_encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False

        # JEPA predictor (narrow bottleneck)
        self.predictor = Predictor(config.encoder.embed_dim, config.predictor)

        # MLM head
        self.mlm_head = MLMHead(
            config.encoder.embed_dim,
            config.encoder.vocab_size,
            config.encoder.norm_type,
        )

        # Optional fragment predictor
        self.fragment_pred = None
        if config.loss.fragment.enabled:
            self.fragment_pred = FragmentPredictor(
                config.encoder.embed_dim, config.loss.fragment
            )

        self._target_mode = config.loss.target_mode

    @torch.no_grad()
    def update_target_encoder(self, decay: Optional[float] = None) -> None:
        """Update target encoder from context encoder.

        If target_mode == "stop_grad": full copy (decay ignored).
        If target_mode == "ema": exponential moving average with given decay.
        """
        if self._target_mode == "stop_grad" or decay is None:
            # Full copy — equivalent to stop-grad with synced weights
            for tp, cp in zip(self.target_encoder.parameters(), self.context_encoder.parameters()):
                tp.data.copy_(cp.data)
        else:
            # EMA update
            for tp, cp in zip(self.target_encoder.parameters(), self.context_encoder.parameters()):
                tp.data.mul_(decay).add_(cp.data, alpha=1.0 - decay)

    @staticmethod
    def get_ema_decay(step: int, total_steps: int, start: float = 0.996, end: float = 1.0) -> float:
        """Cosine EMA schedule: start -> end over training."""
        progress = step / max(total_steps, 1)
        return end - (end - start) * (1.0 + math.cos(math.pi * progress)) / 2.0

    def forward(
        self,
        tokens: torch.Tensor,
        masked_tokens: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass for pretraining.

        Args:
            tokens: (B, L) original token ids (for target encoder)
            masked_tokens: (B, L) masked token ids (for context encoder)
            attention_mask: (B, L) bool, True = valid token

        Returns:
            dict with keys:
                mlm_logits:   (B, L, vocab_size)
                jepa_pred:    (B, D) predicted target CLS
                jepa_target:  (B, D) actual target CLS (detached)
                context_cls:  (B, D) context encoder CLS
                target_cls:   (B, D) target encoder CLS
        """
        # Context encoder on masked input
        context_out = self.context_encoder(masked_tokens, attention_mask, return_all_tokens=True)
        context_cls = context_out["cls"]       # (B, D)
        context_tokens = context_out["tokens"]  # (B, L+1, D)

        # Target encoder on full input (no gradient)
        with torch.no_grad():
            target_out = self.target_encoder(tokens, attention_mask, return_all_tokens=False)
            target_cls = target_out["cls"]     # (B, D)

        # JEPA prediction: predict target CLS from context CLS
        jepa_pred = self.predictor(context_cls)

        # MLM logits from context encoder token outputs (skip CLS at position 0)
        mlm_logits = self.mlm_head(context_tokens[:, 1:, :])  # (B, L, vocab_size)

        return {
            "mlm_logits": mlm_logits,
            "jepa_pred": jepa_pred,
            "jepa_target": target_cls.detach(),
            "context_cls": context_cls,
            "target_cls": target_cls.detach(),
        }

    def forward_fragment(
        self,
        fragment_tokens: torch.Tensor,
        fragment_mask: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """Fragment-level JEPA forward pass.

        Args:
            fragment_tokens: (B, K, L) tokens for K fragments from same genome
            fragment_mask: (B, K, L) attention masks

        Returns:
            dict with fragment_pred and fragment_target
        """
        if self.fragment_pred is None:
            raise RuntimeError("Fragment predictor not enabled in config")

        B, K, L = fragment_tokens.shape

        # Encode all fragments
        flat_tokens = fragment_tokens.reshape(B * K, L)
        flat_mask = fragment_mask.reshape(B * K, L) if fragment_mask is not None else None

        with torch.no_grad():
            target_out = self.target_encoder(flat_tokens, flat_mask)
            all_cls = target_out["cls"].reshape(B, K, -1)  # (B, K, D)

        # Hold out last fragment as target, use rest as context
        target_cls = all_cls[:, -1, :]     # (B, D)
        context_cls = all_cls[:, :-1, :]   # (B, K-1, D)

        context_mask = None
        if fragment_mask is not None:
            # Valid if any token in fragment is valid
            context_mask = fragment_mask[:, :-1, :].any(dim=-1)  # (B, K-1)

        # Predict held-out fragment
        fragment_pred = self.fragment_pred(context_cls, context_mask)

        return {
            "fragment_pred": fragment_pred,
            "fragment_target": target_cls.detach(),
        }

    def encode(
        self,
        tokens: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        use_target: bool = True,
    ) -> torch.Tensor:
        """Extract CLS embeddings for downstream tasks.

        Args:
            tokens: (B, L)
            use_target: if True, use target encoder (recommended for eval)
        """
        encoder = self.target_encoder if use_target else self.context_encoder
        with torch.no_grad():
            return encoder.encode(tokens, attention_mask)

    def save_weights(self, path: str, metadata: Optional[dict] = None) -> None:
        """Save model weights with optional metadata."""
        import os
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        payload = {
            "state_dict": self.state_dict(),
            "config": {
                "encoder": self.config.encoder.__dict__,
                "predictor": self.config.predictor.__dict__,
            },
            "metadata": metadata or {},
        }
        torch.save(payload, path)

    def load_weights(self, path: str, strict: bool = True, map_location=None) -> dict:
        """Load model weights."""
        payload = torch.load(path, map_location=map_location, weights_only=False)
        if isinstance(payload, dict) and "state_dict" in payload:
            self.load_state_dict(payload["state_dict"], strict=strict)
            return payload.get("metadata", {})
        self.load_state_dict(payload, strict=strict)
        return {}
