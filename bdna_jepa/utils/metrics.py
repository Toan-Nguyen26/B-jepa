"""Representation quality metrics.

RankMe: Garrido et al., ICML 2023 — effective rank via Shannon entropy of singular values.
"""
from __future__ import annotations

import torch
import numpy as np


def compute_rankme(embeddings: torch.Tensor) -> float:
    """Compute RankMe (effective rank) of embedding matrix.

    RankMe ~= 1 -> collapsed; RankMe -> d -> full rank.
    """
    if embeddings.ndim == 1:
        embeddings = embeddings.unsqueeze(0)
    embeddings = embeddings.float()

    # SVD
    s = torch.linalg.svdvals(embeddings)
    s = s[s > 1e-12]

    if len(s) == 0:
        return 1.0

    # Normalize to probability distribution
    p = s / s.sum()
    # Shannon entropy -> effective rank
    entropy = -(p * torch.log(p + 1e-12)).sum()
    return float(torch.exp(entropy).item())


def compute_feature_std(embeddings: torch.Tensor) -> float:
    """Mean per-feature standard deviation."""
    return float(embeddings.float().std(dim=0).mean().item())


def compute_spectral_analysis(embeddings: torch.Tensor) -> dict:
    """Full SVD analysis: singular values, cumulative variance, power-law alpha."""
    embeddings = embeddings.float()
    s = torch.linalg.svdvals(embeddings)

    # Cumulative variance
    s_sq = s.pow(2)
    total_var = s_sq.sum()
    cumvar = torch.cumsum(s_sq, dim=0) / total_var if total_var > 0 else s_sq

    # Power law exponent (linear fit in log-log space)
    s_nonzero = s[s > 1e-12]
    alpha = 0.0
    if len(s_nonzero) > 2:
        log_rank = np.log(np.arange(1, len(s_nonzero) + 1))
        log_s = np.log(s_nonzero.cpu().numpy())
        # Linear regression: log(s) = -alpha * log(rank) + c
        coeffs = np.polyfit(log_rank, log_s, 1)
        alpha = float(-coeffs[0])

    return {
        "singular_values": s.cpu().numpy(),
        "cumulative_variance": cumvar.cpu().numpy(),
        "power_law_alpha": alpha,
        "effective_rank": compute_rankme(embeddings),
        "feature_std": compute_feature_std(embeddings),
        "top1_explained": float(s_sq[0] / total_var) if total_var > 0 else 0.0,
    }
