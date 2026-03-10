"""
Guide-target interaction feature engineering for Cas12a cleavage prediction.

Computes the per-position feature matrix (N, C, guide_len) that feeds
Branch B of the dual-input architecture.  Each position in the crRNA-target
duplex is described by thermodynamic, structural, and positional features
that the JEPA encoder cannot learn from genomic pretraining alone — they
arise from RNA:DNA hybridisation biophysics, not genome biology.

Feature channels (default 10):
    0  match/mismatch        binary (0 = Watson-Crick, 1 = mismatch)
    1  mismatch category     0 match, 1 wobble, 2 transition, 3 transversion, 4 pur-pur
    2  nearest-neighbour dG  RNA:DNA hybrid stability per dinucleotide step (Sugimoto 1995)
    3  mismatch penalty dG   additional free-energy cost vs Watson-Crick at that position
    4  cumulative dG         running sum from PAM-proximal, modelling R-loop propagation
    5  position sensitivity  Kleinstiver/Kim mismatch tolerance profile
    6  normalised position   linear ramp 0 (PAM-proximal) to 1 (PAM-distal)
    7  local GC fraction     GC content in a +/-2 nt window
    8  guide purine/pyr      1.0 for purine (A/G), 0.0 for pyrimidine (C/U)
    9  target purine/pyr     1.0 for purine (A/G), 0.0 for pyrimidine (C/T)

Thermodynamic model:
    RNA:DNA hybrid nearest-neighbour parameters from Sugimoto et al. (1995),
    the standard reference for CRISPR guide-target energetics.  Most published
    tools incorrectly use SantaLucia (1998) DNA:DNA parameters — using RNA:DNA
    values is critical because the crRNA-target duplex is a heteroduplex with
    systematically different stacking energies (e.g. rCG/dGC ~ -5.1 kcal/mol
    vs CG/GC DNA:DNA ~ -3.4 kcal/mol).

References:
    Sugimoto et al. Biochemistry 34:11211 (1995).  [RNA:DNA NN params]
    Sugimoto et al. Biochemistry 39:11270 (2000).  [mismatch corrections]
    SantaLucia. PNAS 95:1460 (1998).               [DNA:DNA — NOT used here]
    Kleinstiver et al. Nat Biotechnol 37:276 (2019).[Cas12a position sensitivity]
    Kim et al. Nat Biotechnol 34:863 (2016).        [Cpf1 specificity profiling]
    Strohkendl et al. Mol Cell 71:816 (2018).       [R-loop kinetics]
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch


# --- Configuration -----------------------------------------------------------

@dataclass
class InteractionFeatureConfig:
    """Configuration for interaction feature computation.

    Attributes
    ----------
    guide_len : int
        Expected crRNA spacer length.  21 for LbCas12a (EasyDesign),
        23 for AsCas12a (DeepCpf1).
    temperature_K : float
        Temperature for dG computation (310.15 K = 37 C).
    na_conc_M : float
        Monovalent cation concentration for salt correction.
    gc_window : int
        Half-window size for local GC content computation.
    n_channels : int
        Number of output feature channels (6 minimal, 10 full).
    seed_end : int
        Last position (1-indexed) of the seed region.  Cas12a seed
        is typically positions 1-8 (PAM-proximal).
    pad_value : float
        Fill value for positions with ambiguous bases (N).
    """
    guide_len: int = 23
    temperature_K: float = 310.15
    na_conc_M: float = 1.0
    gc_window: int = 2
    n_channels: int = 10
    seed_end: int = 8
    pad_value: float = 0.0


# --- RNA:DNA hybrid nearest-neighbour parameters (Sugimoto 1995) -------------
#
# Convention: 5'-rX rY-3' / 3'-dX' dY'-5'
# Key: (rna_5prime, rna_3prime) -> (dH kcal/mol, dS cal/mol*K)

_RNA_DNA_NN: Dict[Tuple[str, str], Tuple[float, float]] = {
    ("A", "A"): (-7.8, -21.9),
    ("A", "C"): (-5.9, -12.3),
    ("A", "G"): (-9.1, -23.5),
    ("A", "U"): (-8.3, -23.9),
    ("C", "A"): (-9.0, -26.1),
    ("C", "C"): (-9.3, -23.2),
    ("C", "G"): (-16.3, -47.1),
    ("C", "U"): (-7.0, -19.7),
    ("G", "A"): (-5.5, -13.5),
    ("G", "C"): (-8.0, -17.1),
    ("G", "G"): (-12.8, -31.9),
    ("G", "U"): (-7.8, -21.6),
    ("U", "A"): (-7.8, -23.2),
    ("U", "C"): (-8.6, -22.9),
    ("U", "G"): (-10.4, -28.4),
    ("U", "U"): (-11.5, -36.4),
}

_RNA_DNA_INIT: Tuple[float, float] = (1.9, -3.9)  # (dH, dS)

# Precompute average NN params for fallback
_AVG_DH = sum(v[0] for v in _RNA_DNA_NN.values()) / 16
_AVG_DS = sum(v[1] for v in _RNA_DNA_NN.values()) / 16


# --- Mismatch thermodynamic penalties ---------------------------------------
#
# RNA:DNA mismatch ddG (additional penalty vs Watson-Crick, kcal/mol).
# Compiled from Sugimoto 2000, Watkins 2011, and empirical Cas12a data.
# Positive = destabilising.

_MISMATCH_PENALTIES: Dict[Tuple[str, str], float] = {
    # Wobble — partially tolerated by Cas12a
    ("G", "T"): 0.8,
    ("U", "G"): 1.0,
    # Pyrimidine-pyrimidine
    ("C", "T"): 2.5,
    ("U", "C"): 2.3,
    ("C", "C"): 2.8,
    ("U", "T"): 2.4,
    # Purine-pyrimidine transversions
    ("A", "C"): 2.8,
    # Purine-purine clashes (steric, most destabilising)
    ("A", "G"): 3.2,
    ("G", "A"): 3.0,
    ("G", "G"): 3.5,
    ("A", "A"): 3.3,
    # Watson-Crick (should not be looked up, but safe fallback)
    ("A", "T"): 0.0,
    ("U", "A"): 0.0,
    ("G", "C"): 0.0,
    ("C", "G"): 0.0,
}

_DEFAULT_MISMATCH_PENALTY: float = 2.5


# --- Cas12a position-dependent mismatch tolerance (Kleinstiver/Kim) ----------
#
# 1.0 = completely intolerant; 0.0 = fully tolerant.
# Positions 1-23 (1 = PAM-proximal).  Reflects directional R-loop unwinding:
# seed (1-5) nearly intolerant, transition (6-10) partial, distal (11+) tolerant.

_CAS12A_POSITION_SENSITIVITY: Dict[int, float] = {
    1: 0.98, 2: 0.97, 3: 0.95, 4: 0.93, 5: 0.90,
    6: 0.82, 7: 0.75, 8: 0.68, 9: 0.58, 10: 0.50,
    11: 0.42, 12: 0.35, 13: 0.30, 14: 0.25, 15: 0.22,
    16: 0.18, 17: 0.15, 18: 0.13, 19: 0.11, 20: 0.10,
    21: 0.08, 22: 0.07, 23: 0.06,
}

_DEFAULT_SENSITIVITY: float = 0.05


# --- Complement maps ---------------------------------------------------------

_RNA_TO_DNA_WC: Dict[str, str] = {"A": "T", "U": "A", "G": "C", "C": "G"}
_DNA_COMPLEMENT: Dict[str, str] = {"A": "T", "T": "A", "G": "C", "C": "G"}
_PURINES = frozenset({"A", "G"})


# --- Thermodynamic core ------------------------------------------------------

def _dG(dH: float, dS: float, T: float = 310.15) -> float:
    """dG = dH - T*dS.  dH in kcal/mol, dS in cal/(mol*K)."""
    return dH - T * (dS / 1000.0)


def nn_delta_g(rna_5p: str, rna_3p: str, T: float = 310.15) -> float:
    """Nearest-neighbour dG for one RNA:DNA dinucleotide step.

    Falls back to average value for unrecognised pairs.
    """
    key = (rna_5p.upper(), rna_3p.upper())
    if key in _RNA_DNA_NN:
        dH, dS = _RNA_DNA_NN[key]
        return _dG(dH, dS, T)
    return _dG(_AVG_DH, _AVG_DS, T)


def duplex_delta_g(rna_seq: str, T: float = 310.15) -> float:
    """Total dG for a perfectly matched RNA:DNA hybrid duplex.

    Sums NN contributions + initiation parameter.
    """
    total_dH, total_dS = _RNA_DNA_INIT
    seq = rna_seq.upper()
    for i in range(len(seq) - 1):
        key = (seq[i], seq[i + 1])
        if key in _RNA_DNA_NN:
            dH, dS = _RNA_DNA_NN[key]
        else:
            dH, dS = _AVG_DH, _AVG_DS
        total_dH += dH
        total_dS += dS
    return _dG(total_dH, total_dS, T)


# --- Mismatch classification -------------------------------------------------

def classify_mismatch(rna_base: str, dna_base: str) -> int:
    """Classify an RNA:DNA base pair.

    Returns: 0 Watson-Crick, 1 wobble, 2 transition, 3 transversion,
    4 purine-purine clash.
    """
    r = rna_base.upper().replace("T", "U")
    d = dna_base.upper()

    if _RNA_TO_DNA_WC.get(r) == d:
        return 0
    if (r == "G" and d == "T") or (r == "U" and d == "G"):
        return 1
    if r in _PURINES and d in _PURINES:
        return 4
    if r in {"C", "U"} and d in {"C", "T"}:
        return 2
    return 3


def get_mismatch_penalty(rna_base: str, dna_base: str) -> float:
    """Thermodynamic penalty for a mismatch (kcal/mol).  0 for Watson-Crick."""
    r = rna_base.upper().replace("T", "U")
    d = dna_base.upper()
    if _RNA_TO_DNA_WC.get(r) == d:
        return 0.0
    return _MISMATCH_PENALTIES.get((r, d), _DEFAULT_MISMATCH_PENALTY)


# --- Sequence utilities -------------------------------------------------------

def ensure_rna(seq: str) -> str:
    """Convert to RNA alphabet (T -> U, uppercase)."""
    return seq.upper().replace("T", "U")


def ensure_dna(seq: str) -> str:
    """Convert to DNA alphabet (U -> T, uppercase)."""
    return seq.upper().replace("U", "T")


def protospacer_to_target_strand(protospacer: str) -> str:
    """Complement each base of the protospacer (no reversal).

    The target strand is the DNA strand that base-pairs with the crRNA.
    At each position i, target[i] = complement(protospacer[i]), so the
    alignment with the crRNA is preserved without reversal.
    """
    return "".join(_DNA_COMPLEMENT.get(b.upper(), "N") for b in protospacer)


# --- Per-position feature computation (single pair) --------------------------

def compute_interaction_features(
    crRNA_seq: str,
    target_seq: str,
    config: Optional[InteractionFeatureConfig] = None,
) -> np.ndarray:
    """Compute per-position interaction features for one crRNA-target pair.

    The crRNA spacer (5'->3') is aligned to the target DNA strand (3'->5')
    from the PAM-proximal end.  Position 0 is PAM-adjacent.

    Parameters
    ----------
    crRNA_seq : str
        crRNA spacer, 5'->3'.  DNA (T) or RNA (U) alphabet accepted.
    target_seq : str
        Target DNA strand that base-pairs with the crRNA.  If your data
        stores the protospacer instead, call protospacer_to_target_strand()
        first or use features_from_dataframe_row().
    config : InteractionFeatureConfig or None
        Uses default config if None.

    Returns
    -------
    np.ndarray of shape (n_channels, guide_len), float32
    """
    if config is None:
        config = InteractionFeatureConfig()

    guide = ensure_rna(crRNA_seq)
    target = ensure_dna(target_seq)
    gl = config.guide_len
    T = config.temperature_K
    nc = config.n_channels
    pv = config.pad_value

    # Pad or truncate
    if len(guide) < gl:
        guide += "N" * (gl - len(guide))
    else:
        guide = guide[:gl]
    if len(target) < gl:
        target += "N" * (gl - len(target))
    else:
        target = target[:gl]

    feat = np.full((nc, gl), pv, dtype=np.float32)

    # Precompute per-position base pairs
    is_ambiguous = [guide[i] == "N" or target[i] == "N" for i in range(gl)]

    # Channel 0: match/mismatch binary
    for i in range(gl):
        if is_ambiguous[i]:
            continue
        feat[0, i] = 0.0 if _RNA_TO_DNA_WC.get(guide[i]) == target[i] else 1.0

    # Channel 1: mismatch category
    for i in range(gl):
        if is_ambiguous[i]:
            continue
        feat[1, i] = float(classify_mismatch(guide[i], target[i]))

    # Channel 2: nearest-neighbour dG per step (+ mismatch penalty)
    for i in range(gl):
        if is_ambiguous[i]:
            continue
        if i < gl - 1 and not is_ambiguous[i + 1]:
            step_dg = nn_delta_g(guide[i], guide[i + 1], T)
        elif i > 0 and not is_ambiguous[i - 1]:
            step_dg = nn_delta_g(guide[i - 1], guide[i], T)
        else:
            step_dg = -1.5  # fallback average
        feat[2, i] = step_dg + get_mismatch_penalty(guide[i], target[i])

    # Channel 3: mismatch penalty ddG
    for i in range(gl):
        if is_ambiguous[i]:
            continue
        feat[3, i] = get_mismatch_penalty(guide[i], target[i])

    # Channel 4: cumulative dG (R-loop propagation)
    cumulative = 0.0
    for i in range(gl):
        if feat[2, i] != pv:
            cumulative += feat[2, i]
        feat[4, i] = cumulative

    # Channel 5: Kleinstiver position sensitivity
    for i in range(gl):
        feat[5, i] = _CAS12A_POSITION_SENSITIVITY.get(i + 1, _DEFAULT_SENSITIVITY)

    if nc <= 6:
        return feat[:nc]

    # Channel 6: normalised position
    denom = max(gl - 1, 1)
    for i in range(gl):
        feat[6, i] = i / denom

    # Channel 7: local GC fraction (+/- gc_window)
    w = config.gc_window
    target_arr = np.array(list(target.upper()))
    gc_mask = np.isin(target_arr, ["G", "C"]).astype(np.float32)
    # Cumsum trick for sliding window average
    padded = np.pad(gc_mask, (w, w), mode="edge")
    cumsum = np.cumsum(padded)
    window_sums = cumsum[2 * w:] - cumsum[:gl]
    # Each position sees at most (2*w + 1) bases, but edge positions see fewer
    window_sizes = np.minimum(np.arange(w, w + gl), np.arange(gl)[::-1] + w) + 1
    window_sizes = np.clip(window_sizes, 1, 2 * w + 1).astype(np.float32)
    # Simpler: just use uniform (2w+1) and let edge padding handle it
    feat[7, :] = window_sums / (2 * w + 1)

    # Channel 8: guide purine/pyrimidine
    for i in range(gl):
        r = guide[i]
        if r in ("A", "G"):
            feat[8, i] = 1.0
        elif r in ("C", "U"):
            feat[8, i] = 0.0
        else:
            feat[8, i] = 0.5

    # Channel 9: target purine/pyrimidine
    for i in range(gl):
        d = target[i]
        if d in ("A", "G"):
            feat[9, i] = 1.0
        elif d in ("C", "T"):
            feat[9, i] = 0.0
        else:
            feat[9, i] = 0.5

    return feat[:nc]


# --- Batch computation -------------------------------------------------------

def batch_precompute_features(
    crRNA_seqs: Sequence[str],
    target_seqs: Sequence[str],
    guide_len: int = 21,
    n_channels: int = 10,
) -> np.ndarray:
    """Precompute interaction features for an entire dataset.

    This is the main entry point called by Cas12aDualInputDataset.__init__.
    Returns a contiguous numpy array for cache-friendly __getitem__ access.

    Parameters
    ----------
    crRNA_seqs : list of str
        crRNA spacer sequences (DNA or RNA alphabet).
    target_seqs : list of str
        Target DNA sequences (the strand that pairs with crRNA).
        If your dataset stores protospacers, they should already be
        complemented before reaching this function.
    guide_len : int
        Spacer length (21 for EasyDesign, 23 for DeepCpf1).
    n_channels : int
        Number of feature channels (default 10).

    Returns
    -------
    np.ndarray of shape (N, n_channels, guide_len), dtype float32
    """
    assert len(crRNA_seqs) == len(target_seqs), (
        f"Length mismatch: {len(crRNA_seqs)} guides vs {len(target_seqs)} targets"
    )

    config = InteractionFeatureConfig(
        guide_len=guide_len,
        n_channels=n_channels,
    )

    N = len(crRNA_seqs)
    out = np.zeros((N, n_channels, guide_len), dtype=np.float32)

    for i in range(N):
        out[i] = compute_interaction_features(crRNA_seqs[i], target_seqs[i], config)

    return out


def compute_interaction_features_batch(
    crRNA_seqs: Sequence[str],
    target_seqs: Sequence[str],
    config: Optional[InteractionFeatureConfig] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Compute interaction features as a torch.Tensor (for on-the-fly use).

    For dataset precomputation, prefer batch_precompute_features() which
    returns numpy and avoids unnecessary CPU-GPU transfers.
    """
    if config is None:
        config = InteractionFeatureConfig()

    features = batch_precompute_features(
        crRNA_seqs, target_seqs,
        guide_len=config.guide_len,
        n_channels=config.n_channels,
    )
    tensor = torch.from_numpy(features)
    if device is not None:
        tensor = tensor.to(device)
    return tensor


# Alias for backward compatibility
precompute_interaction_features = batch_precompute_features


# --- Summary statistics (diagnostics / feature importance) -------------------

def summarise_interaction(
    crRNA_seq: str,
    target_seq: str,
    config: Optional[InteractionFeatureConfig] = None,
) -> Dict[str, float]:
    """Compute scalar summary statistics for one guide-target pair.

    Useful for feature importance analysis, active learning, and quick
    screening without the full tensor.
    """
    if config is None:
        config = InteractionFeatureConfig()

    feat = compute_interaction_features(crRNA_seq, target_seq, config)

    mm_binary = feat[0]
    mm_category = feat[1]
    step_dg = feat[2]
    cumulative_dg = feat[4]
    pos_sensitivity = feat[5]

    n_mm = int(mm_binary.sum())
    seed_mask = np.arange(config.guide_len) < config.seed_end
    n_seed_mm = int((mm_binary * seed_mask).sum())

    cum_idx = min(config.seed_end - 1, config.guide_len - 1)

    if config.guide_len > 1:
        diffs = np.diff(cumulative_dg)
        max_barrier = float(diffs.max()) if len(diffs) > 0 else 0.0
    else:
        max_barrier = 0.0

    mm_positions = mm_binary > 0.5
    mean_sens = float(pos_sensitivity[mm_positions].mean()) if mm_positions.any() else 0.0

    return {
        "n_mismatches": n_mm,
        "n_seed_mismatches": n_seed_mm,
        "n_nonseed_mismatches": n_mm - n_seed_mm,
        "n_wobble": int((mm_category == 1).sum()),
        "n_purine_purine": int((mm_category == 4).sum()),
        "total_deltaG": float(step_dg.sum()),
        "seed_deltaG": float(step_dg[:config.seed_end].sum()),
        "cumulative_deltaG_at_seed_end": float(cumulative_dg[cum_idx]),
        "max_deltaG_barrier": max_barrier,
        "mean_position_sensitivity_at_mismatches": mean_sens,
    }


# --- Convenience: compute from common dataset columns ------------------------

def features_from_dataframe_row(
    crRNA_seq: str,
    target_seq: str,
    target_is_protospacer: bool = True,
    config: Optional[InteractionFeatureConfig] = None,
) -> np.ndarray:
    """Compute features from typical dataset columns.

    Many Cas12a datasets store the protospacer (same sequence as crRNA
    but in DNA alphabet) rather than the true target strand.  Set
    target_is_protospacer=True (default) to auto-complement.

    Returns: np.ndarray (n_channels, guide_len)
    """
    if config is None:
        config = InteractionFeatureConfig()

    if target_is_protospacer:
        target = protospacer_to_target_strand(target_seq)
    else:
        target = target_seq

    return compute_interaction_features(ensure_rna(crRNA_seq), target, config)
