"""
PyTorch datasets for DNA-Bacteria-JEPA pretraining and Cas12a fine-tuning.

Supports two Cas12a activity datasets (auto-detected from CSV columns):

  ┌─────────────┬───────────────────┬────────────────────────────────────┐
  │ Dataset     │ DeepCpf1          │ EasyDesign                        │
  │ Reference   │ Kim et al. 2018   │ Huang et al. 2024                 │
  │             │ Nat Biotechnol    │ iMeta 3:e214                      │
  ├─────────────┼───────────────────┼────────────────────────────────────┤
  │ Nuclease    │ AsCas12a          │ LbCas12a                          │
  │ Readout     │ cis-cleavage      │ trans-cleavage (DETECTR)          │
  │             │ (genome editing)  │ (collateral ssDNA fluorescence)   │
  │ Spacer      │ 23 bp             │ 21 bp                             │
  │ PAM         │ TTTV              │ TTTN                              │
  │ Label       │ efficiency_norm   │ fluorescence_log (train)          │
  │             │ [0, 1]            │ fluorescence_raw (test)           │
  │ Split       │ random            │ predefined S3 (train) / S5 (test) │
  │ N           │ ~20K              │ ~11K                              │
  └─────────────┴───────────────────┴────────────────────────────────────┘

Label normalisation strategy (critical for cross-scale evaluation):
  - EasyDesign train/val: z-score on fluorescence_log, stats from train only
  - EasyDesign test: fluorescence_raw kept as-is (Spearman rho is rank-invariant,
    so monotonic transforms do not affect the primary evaluation metric)
  - DeepCpf1: efficiency_normalized used directly (already [0, 1])

References:
  - Kim et al. (2018). Nat Biotechnol 36:238-241.
  - Huang et al. (2024). iMeta 3:e214.
"""
from __future__ import annotations

import copy
import re
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


# ──────────────────────────────────────────────────────────────────────
# Tokenisation
# ──────────────────────────────────────────────────────────────────────

NUCLEOTIDE_VOCAB: Dict[str, int] = {
    "<PAD>": 0,
    "A": 1,
    "C": 2,
    "G": 3,
    "T": 4,
    "<UNK>": 5,
}
VOCAB_SIZE: int = len(NUCLEOTIDE_VOCAB)

_VALID_NT = re.compile(r"^[ACGTacgt]+$")


def tokenize_sequence(seq: str, max_len: int) -> List[int]:
    """Convert a DNA sequence to integer token IDs with right-padding.

    Parameters
    ----------
    seq : str
        DNA string (case-insensitive). Non-ACGT characters map to <UNK>.
    max_len : int
        Fixed output length. Sequences shorter than max_len are padded
        with <PAD> (0); longer sequences are truncated.

    Returns
    -------
    List[int]
        Token IDs of length exactly max_len.
    """
    unk = NUCLEOTIDE_VOCAB["<UNK>"]
    pad = NUCLEOTIDE_VOCAB["<PAD>"]
    tokens = [NUCLEOTIDE_VOCAB.get(nt.upper(), unk) for nt in seq]
    n = len(tokens)
    if n < max_len:
        tokens.extend([pad] * (max_len - n))
    elif n > max_len:
        tokens = tokens[:max_len]
    return tokens


def validate_sequence(seq: str, expected_len: Optional[int] = None) -> bool:
    """Check that a sequence contains only valid nucleotides."""
    if not isinstance(seq, str) or len(seq) == 0:
        return False
    if not _VALID_NT.match(seq):
        return False
    if expected_len is not None and len(seq) != expected_len:
        return False
    return True


# ──────────────────────────────────────────────────────────────────────
# Dataset format auto-detection
# ──────────────────────────────────────────────────────────────────────

class DatasetFormat:
    """Detected dataset configuration descriptor.

    Class-level constants DEEPCPF1 and EASYDESIGN are used as sentinel
    values for format comparison throughout the codebase.
    """

    DEEPCPF1: str = "deepcpf1"
    EASYDESIGN: str = "easydesign"

    def __init__(
        self,
        name: str,
        spacer_len: int,
        label_col: str,
        has_predefined_split: bool,
    ) -> None:
        self.name = name
        self.spacer_len = spacer_len
        self.label_col = label_col
        self.has_predefined_split = has_predefined_split

    def __repr__(self) -> str:
        return (
            f"DatasetFormat(name={self.name!r}, spacer_len={self.spacer_len}, "
            f"label_col={self.label_col!r}, split={self.has_predefined_split})"
        )


def detect_format(df: pd.DataFrame) -> DatasetFormat:
    """Auto-detect dataset format from CSV column names.

    Raises ValueError with actionable message if format is unrecognised.
    """
    cols = set(df.columns)

    if "fluorescence_log" in cols and "fluorescence_raw" in cols:
        return DatasetFormat(
            name=DatasetFormat.EASYDESIGN,
            spacer_len=21,
            label_col="fluorescence_log",
            has_predefined_split=True,
        )
    elif "efficiency_normalized" in cols:
        return DatasetFormat(
            name=DatasetFormat.DEEPCPF1,
            spacer_len=23,
            label_col="efficiency_normalized",
            has_predefined_split=False,
        )
    else:
        raise ValueError(
            f"Cannot detect dataset format from columns: {sorted(cols)}.\n"
            "Expected one of:\n"
            "  - DeepCpf1:   'efficiency_normalized' column\n"
            "  - EasyDesign: 'fluorescence_log' + 'fluorescence_raw' columns\n"
            "Run the appropriate parser first:\n"
            "  python scripts/parse_deepcpf1.py\n"
            "  python scripts/parse_easydesign.py"
        )


# ──────────────────────────────────────────────────────────────────────
# Pretraining dataset
# ──────────────────────────────────────────────────────────────────────

class BacterialGenomeDataset(Dataset):
    """Dataset for JEPA pretraining on bacterial genome fragments.

    Expects CSV with columns: sequence (str), gc_content (float), species (str).
    """

    def __init__(self, csv_path: str, max_tokens: int = 512) -> None:
        self.df = pd.read_csv(csv_path)
        self.max_tokens = max_tokens
        if "sequence" not in self.df.columns:
            raise ValueError(
                f"CSV must have a 'sequence' column. Found: {list(self.df.columns)}"
            )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        tokens = tokenize_sequence(row["sequence"], self.max_tokens)
        out: Dict[str, torch.Tensor] = {
            "tokens": torch.tensor(tokens, dtype=torch.long),
        }
        if "gc_content" in self.df.columns:
            out["gc_content"] = torch.tensor(row["gc_content"], dtype=torch.float32)
        return out


# ──────────────────────────────────────────────────────────────────────
# Cas12a activity dataset
# ──────────────────────────────────────────────────────────────────────

class Cas12aDataset(Dataset):
    """Unified Cas12a dataset supporting both DeepCpf1 and EasyDesign.

    Tokenises the target sequence (PAM + protospacer context) as input
    to the JEPA encoder.  The label column and normalisation strategy
    are determined automatically from the detected format.

    Label normalisation (prevents information leakage):
      - EasyDesign train/val: z-score normalisation using TRAINING set
        statistics only.  The (mean, std) pair is computed once at
        train-set construction and propagated to val/test via the
        label_mean / label_std constructor arguments.
      - EasyDesign test: fluorescence_raw is used directly.  Since
        Spearman rho (the primary metric) is rank-based, any monotonic
        transform of predictions preserves it.
      - DeepCpf1: efficiency_normalized is already bounded [0, 1].

    Parameters
    ----------
    csv_path : str
        Path to the parsed CSV.
    split : {"train", "val", "test", None}
        Which split to load.  For EasyDesign, "train"/"val" select
        S3 rows and "test" selects S5 rows.  None loads everything.
    max_tokens : int
        Maximum token sequence length (including PAM prefix).
    label_mean, label_std : float or None
        Z-score normalisation parameters.  If None and the format
        requires normalisation, they are computed from the loaded data.
        For val/test sets, pass the training set values to prevent leakage.
    """

    def __init__(
        self,
        csv_path: str,
        split: Optional[str] = None,
        max_tokens: int = 512,
        label_mean: Optional[float] = None,
        label_std: Optional[float] = None,
    ) -> None:
        full_df = pd.read_csv(csv_path)
        self.fmt = detect_format(full_df)
        self.max_tokens = max_tokens
        self.split = split

        # ── Row selection by split ──────────────────────────────────
        if split is not None and self.fmt.has_predefined_split:
            if split in ("train", "val"):
                df = full_df[full_df["source"].str.startswith("S3_")].copy()
            elif split == "test":
                df = full_df[full_df["source"].str.startswith("S5_")].copy()
            else:
                raise ValueError(
                    f"Unknown split '{split}'. Expected 'train', 'val', or 'test'."
                )
        else:
            df = full_df.copy()

        # ── Label selection and NaN filtering ───────────────────────
        self.use_raw_labels = False
        if self.fmt.name == DatasetFormat.EASYDESIGN:
            if split == "test":
                df = df.dropna(subset=["fluorescence_raw"])
                self.use_raw_labels = True
            else:
                df = df.dropna(subset=["fluorescence_log"])
        else:
            df = df.dropna(subset=[self.fmt.label_col])

        df = df.reset_index(drop=True)

        # ── Sequence validation (warn, don't crash) ─────────────────
        if "target_seq" in df.columns:
            valid_mask = df["target_seq"].apply(
                lambda s: validate_sequence(str(s), expected_len=self.fmt.spacer_len)
            )
            n_invalid = (~valid_mask).sum()
            if n_invalid > 0:
                warnings.warn(
                    f"{n_invalid}/{len(df)} samples have invalid target_seq "
                    f"(expected {self.fmt.spacer_len}bp ACGT). Dropping them.",
                    stacklevel=2,
                )
                df = df[valid_mask].reset_index(drop=True)

        # ── Z-score normalisation ───────────────────────────────────
        self.label_mean = label_mean
        self.label_std = label_std

        if self.fmt.name == DatasetFormat.EASYDESIGN and not self.use_raw_labels:
            if self.label_mean is None:
                self.label_mean = float(df["fluorescence_log"].mean())
                self.label_std = float(df["fluorescence_log"].std())
            std_safe = max(self.label_std, 1e-8)
            df["_label"] = (df["fluorescence_log"] - self.label_mean) / std_safe
        elif self.fmt.name == DatasetFormat.EASYDESIGN and self.use_raw_labels:
            df["_label"] = df["fluorescence_raw"]
        else:
            df["_label"] = df[self.fmt.label_col]

        self.df = df
        self.spacer_len = self.fmt.spacer_len

        # ── Summary ─────────────────────────────────────────────────
        sources = sorted(df["source"].unique()) if "source" in df.columns else ["unknown"]
        print(
            f"Cas12aDataset [{self.fmt.name}] split={split}: {len(df)} samples "
            f"— sources: {sources}, spacer_len={self.spacer_len}"
        )

    def __len__(self) -> int:
        return len(self.df)

    def _build_input_sequence(self, row: pd.Series) -> str:
        """Build encoder input: PAM prefix + target protospacer.

        The PAM is prepended so the encoder sees the full recognition
        context.  For EasyDesign (TTTN PAM, 21bp spacer) the total
        input is ~25nt; for DeepCpf1 (TTTV PAM, 23bp spacer) ~27nt.
        """
        pam = str(row.get("PAM", "TTTN"))
        target = str(row["target_seq"])
        return pam + target

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        seq = self._build_input_sequence(row)
        tokens = tokenize_sequence(seq, self.max_tokens)

        return {
            "tokens": torch.tensor(tokens, dtype=torch.long),
            "label": torch.tensor(float(row["_label"]), dtype=torch.float32),
        }

    def get_normalisation_params(self) -> Tuple[Optional[float], Optional[float]]:
        """Return (mean, std) used for label z-scoring.

        Pass these values to val/test Cas12aDataset constructors to
        prevent label leakage across splits.
        """
        return self.label_mean, self.label_std


# ──────────────────────────────────────────────────────────────────────
# Dual-input dataset wrapper (Branch A + Branch B)
# ──────────────────────────────────────────────────────────────────────

class Cas12aDualInputDataset(Dataset):
    """Wraps Cas12aDataset with precomputed interaction features for
    the dual-input architecture (Branch A: JEPA context encoder +
    Branch B: 1D CNN over per-position biophysical features).

    Interaction features (10 channels per position):
      1. Match / mismatch binary
      2. Mismatch type (transition / transversion / wobble / purine-purine)
      3. Nearest-neighbour dG (Sugimoto 1995 RNA:DNA hybrid parameters)
      4. Cumulative dG (R-loop propagation profile)
      5. Seed vs non-seed flag (positions 1-8 from PAM)
      6. Normalised position (0 -> 1 from PAM-proximal to distal)
      7. Kleinstiver positional sensitivity weights
      8. Mismatch penalty (category x position interaction)
      9. Local GC content (sliding window)
     10. Secondary structure propensity

    Features are precomputed once at __init__ and cached as a single
    (N, n_channels, spacer_len) numpy array.  __getitem__ is a pure
    index + .copy() — zero thermodynamic computation during training.

    Returns per sample:
      tokens:               (max_tokens,)              — JEPA encoder input
      interaction_features:  (n_channels, spacer_len)   — Branch B CNN input
      label:                 scalar                     — activity label
    """

    def __init__(self, base_dataset: Cas12aDataset) -> None:
        self.base = base_dataset
        self.spacer_len = base_dataset.spacer_len
        self.df = base_dataset.df  # expose for raw_targets / source_labels access
        self.label_mean = base_dataset.label_mean
        self.label_std = base_dataset.label_std

        # Lazy import to avoid circular dependency (features.py imports nothing
        # from dataset.py, but keeping the import local is defensive).
        from src.cas12a.features import batch_precompute_features

        print(
            f"  Precomputing interaction features for {len(self.base)} samples "
            f"(spacer_len={self.spacer_len})..."
        )

        crRNAs = self.base.df["crRNA_seq"].tolist()
        targets = self.base.df["target_seq"].tolist()

        self.feature_cache: np.ndarray = batch_precompute_features(
            crRNAs, targets, guide_len=self.spacer_len
        )
        n_channels = self.feature_cache.shape[1]
        mem_mb = self.feature_cache.nbytes / 1e6
        print(
            f"  Cached: ({len(self.base)}, {n_channels}, {self.spacer_len}) "
            f"float32, {mem_mb:.1f} MB"
        )

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.base[idx]
        item["interaction_features"] = torch.from_numpy(
            self.feature_cache[idx].copy()
        ).float()
        return item

    def get_normalisation_params(self) -> Tuple[Optional[float], Optional[float]]:
        """Delegate to underlying base dataset."""
        return self.base.get_normalisation_params()


# ──────────────────────────────────────────────────────────────────────
# Train / val / test splitting
# ──────────────────────────────────────────────────────────────────────

def _make_subset(parent: Cas12aDataset, sub_df: pd.DataFrame) -> Cas12aDataset:
    """Create a shallow copy of a Cas12aDataset with a replacement DataFrame.

    Preserves format, normalisation params, and tokenisation settings.
    """
    ds = copy.copy(parent)
    ds.df = sub_df.reset_index(drop=True)
    return ds


def build_splits(
    csv_path: str,
    max_tokens: int = 512,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
) -> Tuple[Cas12aDataset, Cas12aDataset, Cas12aDataset]:
    """Build train / val / test datasets from a single CSV.

    Splitting strategy depends on the detected format:

    **EasyDesign** (predefined split):
      S3 rows -> train + val (randomly split by val_frac within S3).
      S5 rows -> test (held-out, never seen during training).
      Label normalisation: z-score stats computed on train split only,
      then propagated to val (and test, though test uses raw labels).

    **DeepCpf1** (random split):
      Random split using val_frac + test_frac with fixed seed.
      No label normalisation (efficiency_normalized is already [0, 1]).

    Parameters
    ----------
    csv_path : str
        Path to the parsed dataset CSV.
    max_tokens : int
        Max token length for JEPA encoder input.
    val_frac : float
        Fraction of training data reserved for validation.
    test_frac : float
        Fraction for test set (DeepCpf1 only; EasyDesign uses S5).
    seed : int
        Random seed for reproducible splitting.

    Returns
    -------
    (train_ds, val_ds, test_ds) : tuple of Cas12aDataset
    """
    full_df = pd.read_csv(csv_path)
    fmt = detect_format(full_df)

    if fmt.has_predefined_split:
        return _build_easydesign_splits(csv_path, max_tokens, val_frac, seed)
    else:
        return _build_random_splits(csv_path, full_df, fmt, max_tokens,
                                    val_frac, test_frac, seed)


def _build_easydesign_splits(
    csv_path: str,
    max_tokens: int,
    val_frac: float,
    seed: int,
) -> Tuple[Cas12aDataset, Cas12aDataset, Cas12aDataset]:
    """EasyDesign: S3 -> train+val, S5 -> test."""
    # Load full S3 to compute normalisation stats
    train_full = Cas12aDataset(csv_path, split="train", max_tokens=max_tokens)
    norm_mean, norm_std = train_full.get_normalisation_params()

    # Random train/val partition within S3
    n = len(train_full.df)
    rng = np.random.RandomState(seed)
    indices = rng.permutation(n)
    n_val = int(n * val_frac)

    val_idx = sorted(indices[:n_val].tolist())
    trn_idx = sorted(indices[n_val:].tolist())

    train_ds = _make_subset(train_full, train_full.df.iloc[trn_idx])
    val_ds = _make_subset(train_full, train_full.df.iloc[val_idx])

    # Test set (S5) — inherits normalisation params for consistency
    test_ds = Cas12aDataset(
        csv_path, split="test", max_tokens=max_tokens,
        label_mean=norm_mean, label_std=norm_std,
    )

    print(f"Splits: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")
    return train_ds, val_ds, test_ds


def _build_random_splits(
    csv_path: str,
    full_df: pd.DataFrame,
    fmt: DatasetFormat,
    max_tokens: int,
    val_frac: float,
    test_frac: float,
    seed: int,
) -> Tuple[Cas12aDataset, Cas12aDataset, Cas12aDataset]:
    """DeepCpf1: random train/val/test split."""
    n = len(full_df)
    rng = np.random.RandomState(seed)
    indices = rng.permutation(n)

    n_test = int(n * test_frac)
    n_val = int(n * val_frac)

    test_idx = set(indices[:n_test].tolist())
    val_idx = set(indices[n_test:n_test + n_val].tolist())

    # Build base dataset (loads all rows, applies format detection)
    base = Cas12aDataset(csv_path, split=None, max_tokens=max_tokens)

    # Partition DataFrame
    mask_test = base.df.index.isin(test_idx)
    mask_val = base.df.index.isin(val_idx)
    mask_train = ~(mask_test | mask_val)

    train_ds = _make_subset(base, base.df[mask_train])
    val_ds = _make_subset(base, base.df[mask_val])
    test_ds = _make_subset(base, base.df[mask_test])

    print(
        f"Splits (random): train={len(train_ds)}, "
        f"val={len(val_ds)}, test={len(test_ds)}"
    )
    return train_ds, val_ds, test_ds


# ──────────────────────────────────────────────────────────────────────
# Collate functions
# ──────────────────────────────────────────────────────────────────────

def cas12a_collate(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Standard collate for single-input (MLP head) mode."""
    return {
        "tokens": torch.stack([b["tokens"] for b in batch]),
        "label": torch.stack([b["label"] for b in batch]),
    }


def dual_input_collate(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Collate for dual-input (Branch A + Branch B) mode.

    Stacks interaction_features alongside tokens and labels.
    """
    out = cas12a_collate(batch)
    if "interaction_features" in batch[0]:
        out["interaction_features"] = torch.stack(
            [b["interaction_features"] for b in batch]
        )
    return out
