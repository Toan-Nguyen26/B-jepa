"""Masking strategies for masked language modeling."""
from __future__ import annotations

import torch


def random_mask(
    tokens: torch.Tensor,
    mask_ratio: float = 0.15,
    mask_id: int = 1,
    vocab_size: int = 128,
    special_token_max: int = 5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Standard BERT-style random masking: 80% [MASK], 10% random, 10% keep.

    Returns:
        masked_tokens: tokens with masking applied
        mask: boolean mask of selected positions
        labels: original token ids at masked positions, -100 elsewhere
    """
    masked_tokens = tokens.clone()
    labels = torch.full_like(tokens, -100)

    # Only mask non-special tokens
    eligible = tokens >= special_token_max
    prob_matrix = torch.full_like(tokens, mask_ratio, dtype=torch.float)
    prob_matrix[~eligible] = 0.0
    mask = torch.bernoulli(prob_matrix).bool()

    labels[mask] = tokens[mask]

    # 80% -> [MASK], 10% -> random, 10% -> keep original
    indices_replaced = torch.bernoulli(torch.full_like(prob_matrix, 0.8)).bool() & mask
    masked_tokens[indices_replaced] = mask_id

    indices_random = torch.bernoulli(torch.full_like(prob_matrix, 0.5)).bool() & mask & ~indices_replaced
    random_tokens = torch.randint(special_token_max, vocab_size, tokens.shape, device=tokens.device)
    masked_tokens[indices_random] = random_tokens[indices_random]

    return masked_tokens, mask, labels


def span_mask(
    tokens: torch.Tensor,
    mask_ratio: float = 0.15,
    span_length: int = 5,
    mask_id: int = 1,
    special_token_max: int = 5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Contiguous span masking — better for DNA local context.

    Selects random span starts and masks contiguous spans of `span_length`.
    """
    B, L = tokens.shape
    masked_tokens = tokens.clone()
    labels = torch.full_like(tokens, -100)
    mask = torch.zeros_like(tokens, dtype=torch.bool)

    n_masked_target = max(1, int(L * mask_ratio))
    n_spans = max(1, n_masked_target // span_length)

    for b in range(B):
        eligible = (tokens[b] >= special_token_max).nonzero(as_tuple=True)[0]
        if len(eligible) < span_length:
            continue

        starts = eligible[torch.randperm(len(eligible), device=tokens.device)[:n_spans]]
        for s in starts:
            end = min(s + span_length, L)
            span_indices = torch.arange(s, end, device=tokens.device)
            # Only mask eligible positions
            valid = tokens[b, span_indices] >= special_token_max
            span_indices = span_indices[valid]
            mask[b, span_indices] = True

    labels[mask] = tokens[mask]
    masked_tokens[mask] = mask_id

    return masked_tokens, mask, labels
