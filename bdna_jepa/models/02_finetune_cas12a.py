"""
Fine-tuning script for Cas12a cleavage activity prediction.

Supports two datasets (auto-detected from CSV columns):
  - DeepCpf1 (Kim et al. 2018): AsCas12a cis-cleavage, 23 bp
  - EasyDesign (Huang et al. 2024): LbCas12a trans-cleavage, 21 bp

Architecture modes (--head):
  mlp    -- 3-layer residual MLP, single-input baseline
  dual   -- Branch A (JEPA context) + Branch B (1D CNN over Sugimoto
            RNA:DNA thermodynamics, Kleinstiver positional sensitivity,
            mismatch taxonomy, R-loop propagation) -> fusion MLP -> scalar
  legacy -- original linear head (ablation baseline)

Training strategy (Howard & Ruder, ACL 2018):
  Phase 1 -- Frozen encoder, head-only (epochs 1 -> --unfreeze-epoch)
  Phase 2 -- Discriminative unfreezing with layer-wise LR decay

Loss: alpha*MSE + beta*(1-CCC) + gamma*Huber
  CCC (Lin 1989) directly optimises rank+magnitude agreement.
  Huber provides robustness to fluorescence measurement outliers.

Evaluation:
  Spearman rho / Pearson r / MAE / RMSE on z-scored labels,
  Spearman rho on raw fluorescence (EasyDesign primary benchmark),
  per-source breakdown, bootstrap 95% CI (1000 resamples).

References:
  Howard & Ruder (2018). ACL. [discriminative LR / gradual unfreezing]
  Loshchilov & Hutter (2017). ICLR. [cosine annealing]
  Lin (1989). Biometrics 45:255. [concordance correlation coefficient]
  Sugimoto et al. (1995). Biochemistry 34:11211. [RNA:DNA NN params]
  Kleinstiver et al. (2019). Nat Biotechnol 37:276. [Cas12a sensitivity]
  Kim et al. (2018). Nat Biotechnol 36:238. [DeepCpf1]
  Huang et al. (2024). iMeta 3:e214. [EasyDesign / CNN12ae]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import random
import sys
import time
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.cas12a.dataset import (
    Cas12aDataset,
    Cas12aDualInputDataset,
    build_splits,
    detect_format,
    cas12a_collate,
    dual_input_collate,
    VOCAB_SIZE,
    DatasetFormat,
)
from src.cas12a.encoder import SparseTransformerEncoder
from src.cas12a.model import (
    Cas12aMLPHead,
    Cas12aDualInputHead,
    Cas12aEfficiencyModel,
    Cas12aDualInputModel,
    DualInputConfig,
)


# --- Composite loss -----------------------------------------------------------

class ConcordanceCorrelationLoss(nn.Module):
    """Lin's concordance correlation coefficient loss (Lin, 1989).

    CCC = 2*cov(x,y) / (var(x) + var(y) + (mu_x - mu_y)^2)

    Minimises (1 - CCC).  Unlike Pearson r, CCC penalises both correlation
    AND calibration: perfectly correlated but shifted predictions still
    incur loss.
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.flatten()
        target = target.flatten()
        mu_p = pred.mean()
        mu_t = target.mean()
        var_p = pred.var(correction=0).clamp(min=self.eps)
        var_t = target.var(correction=0).clamp(min=self.eps)
        cov = ((pred - mu_p) * (target - mu_t)).mean()
        ccc = (2 * cov) / (var_p + var_t + (mu_p - mu_t) ** 2 + self.eps)
        return 1.0 - ccc


class CompositeLoss(nn.Module):
    """alpha*MSE + beta*(1-CCC) + gamma*Huber

    Default weights found by grid search on Cas12a trans-cleavage validation.
    CCC provides gradient signal aligned with Spearman rho optimisation.
    Huber (delta=1.0) dampens fluorescence outlier influence.
    """

    def __init__(self, mse_w: float = 0.5, ccc_w: float = 0.3,
                 huber_w: float = 0.2, huber_delta: float = 1.0):
        super().__init__()
        self.mse_w = mse_w
        self.ccc_w = ccc_w
        self.huber_w = huber_w
        self.mse = nn.MSELoss()
        self.ccc = ConcordanceCorrelationLoss()
        self.huber = nn.HuberLoss(delta=huber_delta)

    def forward(self, pred: torch.Tensor, target: torch.Tensor
                ) -> Tuple[torch.Tensor, Dict[str, float]]:
        pred = pred.squeeze()
        l_mse = self.mse(pred, target)
        l_ccc = self.ccc(pred, target)
        l_hub = self.huber(pred, target)
        total = self.mse_w * l_mse + self.ccc_w * l_ccc + self.huber_w * l_hub
        return total, {
            "loss": total.item(), "mse": l_mse.item(),
            "ccc_loss": l_ccc.item(), "huber": l_hub.item(),
        }


# --- EMA (Polyak averaging, Tarvainen & Valpola 2017) -------------------------

class EMAModel:
    """Exponential moving average of model parameters.

    theta_ema <- decay * theta_ema + (1 - decay) * theta_model
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {
            name: p.clone().detach()
            for name, p in model.named_parameters() if p.requires_grad
        }
        self.backup: Dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for name, p in model.named_parameters():
            if p.requires_grad and name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(p.data, alpha=1.0 - self.decay)

    def apply(self, model: nn.Module) -> None:
        self.backup = {}
        for name, p in model.named_parameters():
            if name in self.shadow:
                self.backup[name] = p.data.clone()
                p.data.copy_(self.shadow[name])

    def restore(self, model: nn.Module) -> None:
        for name, p in model.named_parameters():
            if name in self.backup:
                p.data.copy_(self.backup[name])
        self.backup = {}

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return {k: v.clone() for k, v in self.shadow.items()}

    def load_state_dict(self, sd: Dict[str, torch.Tensor]) -> None:
        self.shadow = {k: v.clone() for k, v in sd.items()}


# --- Utilities ----------------------------------------------------------------

def resolve_path(p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def set_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass


def count_params(model: nn.Module, trainable: bool = True) -> int:
    return sum(p.numel() for p in model.parameters() if (not trainable or p.requires_grad))


def fmt_params(n: int) -> str:
    if n >= 1e6:
        return f"{n / 1e6:.2f}M"
    if n >= 1e3:
        return f"{n / 1e3:.1f}K"
    return str(n)


def select_amp_dtype(precision: str, device: torch.device):
    if device.type != "cuda" or precision == "fp32":
        return None
    if precision == "auto":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if precision == "bf16":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if precision == "fp16":
        return torch.float16
    return None


def get_env_info() -> Dict[str, str]:
    info = {
        "python": sys.version.split()[0],
        "pytorch": torch.__version__,
        "cuda": torch.version.cuda or "N/A",
        "platform": platform.platform(),
    }
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
        info["gpu_memory_gb"] = f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}"
    return info


# --- Discriminative LR (ULMFiT) ----------------------------------------------

def build_discriminative_optimizer(
    model: nn.Module, head_type: str, base_lr: float, weight_decay: float,
    lr_decay: float = 0.8, encoder_lr_scale: float = 0.1,
) -> torch.optim.AdamW:
    """Layer-wise LR: lr_i = base_lr * encoder_lr_scale * decay^(N-1-i)"""
    groups = []
    enc = model.encoder
    n_layers = len(enc.encoder.layers)

    # Embeddings get lowest LR
    embed_lr = base_lr * encoder_lr_scale * (lr_decay ** n_layers)
    embed_params = list(enc.token_embedding.parameters()) + list(enc.pos_embedding.parameters())
    if embed_params:
        groups.append({"params": embed_params, "lr": embed_lr, "name": "encoder/embed"})

    # Transformer layers with increasing LR
    for i, layer in enumerate(enc.encoder.layers):
        layer_lr = base_lr * encoder_lr_scale * (lr_decay ** (n_layers - 1 - i))
        lp = list(layer.parameters())
        if lp:
            groups.append({"params": lp, "lr": layer_lr, "name": f"encoder/layer_{i}"})

    # Task head at full LR
    head = model.task_head
    if head_type == "dual" and hasattr(head, "interaction_cnn"):
        cnn_p = list(head.interaction_cnn.parameters())
        if cnn_p:
            groups.append({"params": cnn_p, "lr": base_lr, "name": "head/cnn"})
        fusion_p = []
        for attr in ("fusion", "output"):
            if hasattr(head, attr):
                fusion_p += list(getattr(head, attr).parameters())
        if fusion_p:
            groups.append({"params": fusion_p, "lr": base_lr, "name": "head/fusion"})
    else:
        hp = list(head.parameters())
        if hp:
            groups.append({"params": hp, "lr": base_lr, "name": "head/all"})

    groups = [g for g in groups if g["params"]]

    print("\n  Discriminative LR schedule:")
    for g in groups:
        n = sum(p.numel() for p in g["params"])
        print(f"    {g['name']:30s}  lr={g['lr']:.2e}  params={fmt_params(n)}")

    return torch.optim.AdamW(groups, lr=base_lr, weight_decay=weight_decay)


# --- LR scheduler -------------------------------------------------------------

class CosineWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    """Linear warmup then cosine decay to min_lr_ratio * base_lr."""

    def __init__(self, optimizer, warmup_steps, total_steps,
                 min_lr_ratio=0.01, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        s = self.last_epoch
        if s < self.warmup_steps:
            scale = (s + 1) / max(self.warmup_steps, 1)
        else:
            prog = (s - self.warmup_steps) / max(self.total_steps - self.warmup_steps, 1)
            prog = min(prog, 1.0)
            scale = self.min_lr_ratio + 0.5 * (1 - self.min_lr_ratio) * (1 + math.cos(math.pi * prog))
        return [lr * scale for lr in self.base_lrs]


# --- Freeze / unfreeze --------------------------------------------------------

def freeze_encoder(m: nn.Module) -> None:
    for p in m.encoder.parameters():
        p.requires_grad = False


def unfreeze_encoder(m: nn.Module) -> None:
    for p in m.encoder.parameters():
        p.requires_grad = True


# --- Bootstrap CI -------------------------------------------------------------

def bootstrap_spearman_ci(
    preds: np.ndarray, targets: np.ndarray,
    n_boot: int = 1000, ci: float = 0.95, seed: int = 42,
) -> Tuple[float, float, float]:
    """Percentile bootstrap CI on Spearman rho (Efron & Tibshirani, 1993)."""
    rng = np.random.RandomState(seed)
    n = len(preds)
    rhos = np.array([spearmanr(preds[rng.choice(n, n, replace=True)],
                               targets[rng.choice(n, n, replace=True)])[0]
                     for _ in range(n_boot)])
    alpha = (1 - ci) / 2
    return (float(spearmanr(preds, targets)[0]),
            float(np.percentile(rhos, 100 * alpha)),
            float(np.percentile(rhos, 100 * (1 - alpha))))


# --- Evaluation ---------------------------------------------------------------

@torch.no_grad()
def evaluate(
    model: nn.Module, loader: DataLoader, device: torch.device,
    use_amp: bool, amp_dtype, head_type: str = "mlp",
    raw_targets: Optional[np.ndarray] = None,
    source_labels: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Full evaluation with optional raw-scale rho and per-source breakdown."""
    model.eval()
    all_preds, all_tgt = [], []

    for batch in loader:
        tokens = batch["tokens"].to(device, non_blocking=True)
        ctx = torch.autocast(device_type="cuda", dtype=amp_dtype) if use_amp else nullcontext()
        with ctx:
            if head_type == "dual" and "interaction_features" in batch:
                inter = batch["interaction_features"].to(device, non_blocking=True)
                preds, _ = model(tokens, inter)
            else:
                preds, _ = model(tokens)
        all_preds.extend(preds.float().cpu().numpy().flatten())
        all_tgt.extend(batch["label"].float().numpy().flatten())

    p = np.array(all_preds)
    t = np.array(all_tgt)
    if len(p) < 2:
        return {"pearson": 0.0, "spearman": 0.0, "mae": 0.0, "rmse": 0.0}

    res = p - t
    result: Dict[str, Any] = {
        "pearson": float(pearsonr(p, t)[0]),
        "spearman": float(spearmanr(p, t)[0]),
        "mae": float(np.abs(res).mean()),
        "rmse": float(np.sqrt((res ** 2).mean())),
        "n_samples": len(p),
        "predictions": p,
        "targets": t,
    }

    # Raw fluorescence Spearman (EasyDesign primary metric)
    if raw_targets is not None and len(raw_targets) == len(p):
        valid = ~np.isnan(raw_targets)
        if valid.sum() > 10:
            result["spearman_raw"] = float(spearmanr(p[valid], raw_targets[valid])[0])

    # Per-source breakdown
    if source_labels is not None and len(source_labels) == len(p):
        per_src = {}
        for src in np.unique(source_labels):
            mask = source_labels == src
            if mask.sum() >= 5:
                per_src[str(src)] = {
                    "n": int(mask.sum()),
                    "spearman": float(spearmanr(p[mask], t[mask])[0]),
                    "pearson": float(pearsonr(p[mask], t[mask])[0]),
                }
        if per_src:
            result["per_source"] = per_src

    return result


# --- Encoder construction with auto vocab detection ---------------------------

def build_encoder(args: argparse.Namespace, pretrained_path: Path
                  ) -> Tuple[SparseTransformerEncoder, Optional[int]]:
    """Build encoder, auto-detecting vocab_size from checkpoint if needed.

    Pretrained JEPA checkpoints may use vocab_size=9 (with <MASK>, <CLS>,
    <SEP> tokens) while the fine-tuning tokenizer uses vocab_size=6.
    Extra embedding rows are dormant during fine-tuning (never indexed)
    but must exist for weight loading to succeed.

    Returns (encoder, pretrain_epoch).
    """
    vocab_size = VOCAB_SIZE
    pretrain_epoch = None

    if pretrained_path.exists():
        ckpt = torch.load(pretrained_path, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("encoder_state_dict", ckpt.get("model_state_dict", ckpt))
        pretrain_epoch = ckpt.get("epoch", None)

        # Detect vocab size from checkpoint embedding shape
        embed_key = "token_embedding.weight"
        if embed_key in state_dict:
            ckpt_vocab = state_dict[embed_key].shape[0]
            if ckpt_vocab != VOCAB_SIZE:
                print(f"  Vocab mismatch: checkpoint={ckpt_vocab}, tokenizer={VOCAB_SIZE}")
                print(f"  Using checkpoint vocab_size={ckpt_vocab} (extra tokens dormant)")
                vocab_size = ckpt_vocab
    else:
        ckpt = None
        state_dict = None

    encoder = SparseTransformerEncoder(
        vocab_size=vocab_size,
        embed_dim=args.embed_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        ff_dim=args.ff_dim,
    )

    if state_dict is not None:
        missing, unexpected = encoder.load_state_dict(state_dict, strict=False)
        if missing:
            trunc = missing[:5]
            print(f"  Warning -- missing keys: {trunc}{'...' if len(missing) > 5 else ''}")
        if unexpected:
            trunc = unexpected[:5]
            print(f"  Warning -- unexpected keys: {trunc}{'...' if len(unexpected) > 5 else ''}")
        print(f"  Loaded pretrained weights (epoch {pretrain_epoch})")
    else:
        print("  Warning: No checkpoint found -- training encoder from scratch")

    return encoder, pretrain_epoch


# --- Main training loop -------------------------------------------------------

def finetune(args: argparse.Namespace) -> None:
    set_seed(args.seed, deterministic=args.deterministic)

    data_path = resolve_path(args.data)
    pretrained_path = resolve_path(args.pretrained_path)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not data_path.exists():
        sys.exit(f"Data not found: {data_path}")

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env_info = get_env_info()
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {env_info.get('gpu', '?')}, {env_info.get('gpu_memory_gb', '?')} GB")
        if not args.deterministic:
            torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    amp_dtype = select_amp_dtype(args.precision, device)
    use_amp = amp_dtype is not None
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and amp_dtype == torch.float16))

    # Dataset
    print(f"\nLoading data from {data_path}")
    probe_df = pd.read_csv(data_path, nrows=5)
    fmt = detect_format(probe_df)
    is_easydesign = fmt.name == DatasetFormat.EASYDESIGN

    if args.guide_len_auto:
        args.guide_len = fmt.spacer_len

    print(f"  Format: {fmt.name}, spacer: {fmt.spacer_len}bp, label: {fmt.label_col}")

    train_ds, val_ds, test_ds = build_splits(
        str(data_path), max_tokens=args.max_tokens,
        val_frac=args.val_fraction, seed=args.seed,
    )
    print(f"  Train: {len(train_ds)}  Val: {len(val_ds)}  Test: {len(test_ds)}")

    # Extract raw targets and source labels for evaluation
    raw_test_targets = raw_val_targets = None
    test_source_labels = val_source_labels = None

    if is_easydesign:
        for ds, name in [(test_ds, "test"), (val_ds, "val")]:
            if "fluorescence_raw" in ds.df.columns:
                raw = ds.df["fluorescence_raw"].values.astype(np.float64)
                if np.isfinite(raw).sum() > 0:
                    if name == "test":
                        raw_test_targets = raw
                    else:
                        raw_val_targets = raw
        if raw_test_targets is not None:
            print(f"  Primary metric: Spearman rho on fluorescence_raw (n_test={np.isfinite(raw_test_targets).sum()})")

    for ds, name in [(test_ds, "test"), (val_ds, "val")]:
        if "source" in ds.df.columns:
            if name == "test":
                test_source_labels = ds.df["source"].values
            else:
                val_source_labels = ds.df["source"].values

    # Wrap for dual-input
    if args.head == "dual":
        print("\nPrecomputing interaction features:")
        train_dataset = Cas12aDualInputDataset(train_ds)
        val_dataset = Cas12aDualInputDataset(val_ds)
        test_dataset = Cas12aDualInputDataset(test_ds)
        collate_fn = dual_input_collate
    else:
        train_dataset, val_dataset, test_dataset = train_ds, val_ds, test_ds
        collate_fn = cas12a_collate

    # Loaders
    lkw: Dict[str, Any] = {
        "batch_size": args.batch_size, "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda", "collate_fn": collate_fn,
    }
    if args.num_workers > 0:
        lkw["persistent_workers"] = True
        lkw["prefetch_factor"] = args.prefetch_factor

    train_loader = DataLoader(train_dataset, shuffle=True, drop_last=True, **lkw)
    val_loader = DataLoader(val_dataset, shuffle=False, **lkw)
    test_loader = DataLoader(test_dataset, shuffle=False, **lkw)

    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = steps_per_epoch * args.warmup_epochs

    # Encoder (with auto vocab detection)
    print(f"\nLoading pretrained encoder from {pretrained_path}")
    encoder, pretrain_epoch = build_encoder(args, pretrained_path)

    # Task head + model
    if args.head == "dual":
        cfg = DualInputConfig(
            embed_dim=args.embed_dim,
            n_interaction_features=args.n_interaction_channels,
            guide_len=args.guide_len,
            dropout=args.head_dropout,
        )
        head = Cas12aDualInputHead(cfg)
        model = Cas12aDualInputModel(encoder, head)
    elif args.head == "mlp":
        head = Cas12aMLPHead(embed_dim=args.embed_dim, dropout=args.head_dropout)
        model = Cas12aEfficiencyModel(encoder, head)
    elif args.head == "legacy":
        from src.cas12a.model import Cas12aEfficiencyHead
        head = Cas12aEfficiencyHead()
        model = Cas12aEfficiencyModel(encoder, head)
    else:
        sys.exit(f"Unknown head: {args.head}")

    model.to(device)
    total_p = count_params(model, trainable=False)
    print(f"\n  Head: {args.head}, guide_len: {args.guide_len}, params: {fmt_params(total_p)}")

    # EMA
    ema: Optional[EMAModel] = None
    if args.ema:
        ema = EMAModel(model, decay=args.ema_decay)
        print(f"  EMA: decay={args.ema_decay}")

    # Freeze encoder (Phase 1)
    encoder_frozen = False
    if args.unfreeze_epoch > 0:
        freeze_encoder(model)
        encoder_frozen = True
        print(f"  Encoder frozen until epoch {args.unfreeze_epoch}")
        print(f"  Trainable (Phase 1): {fmt_params(count_params(model))}")
    else:
        print(f"  Trainable: {fmt_params(count_params(model))}")

    # Optimizer
    if encoder_frozen:
        head_params = [p for p in model.task_head.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(head_params, lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = build_discriminative_optimizer(
            model, args.head, args.lr, args.weight_decay,
            lr_decay=args.lr_decay_factor, encoder_lr_scale=args.encoder_lr_scale,
        )

    scheduler = CosineWarmupScheduler(
        optimizer, warmup_steps=warmup_steps, total_steps=total_steps,
        min_lr_ratio=args.min_lr / args.lr if args.min_lr > 0 else 0.01,
    )

    criterion = CompositeLoss(
        mse_w=args.mse_weight, ccc_w=args.ccc_weight,
        huber_w=args.huber_weight, huber_delta=args.huber_delta,
    )

    grad_accum = max(1, args.grad_accum_steps)

    # W&B
    if args.wandb and HAS_WANDB:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_name or f"cas12a-{args.head}-{fmt.name}-{time.strftime('%m%d-%H%M')}",
            config={**vars(args), **env_info},
            tags=["fine-tune", args.head, fmt.name],
        )

    # Print config
    print(f"\n{'=' * 60}")
    print(f"  Epochs: {args.epochs}, batch: {args.batch_size}x{grad_accum}, steps/ep: {steps_per_epoch}")
    print(f"  Warmup: {args.warmup_epochs}ep ({warmup_steps} steps)")
    print(f"  Loss: {args.mse_weight}*MSE + {args.ccc_weight}*CCC + {args.huber_weight}*Huber(d={args.huber_delta})")
    print(f"  Early stopping: patience={args.patience}")
    if is_easydesign:
        mu, sd = train_ds.get_normalisation_params()
        print(f"  Z-score: mu={mu:.4f}, std={sd:.4f}")
    print(f"{'=' * 60}\n")

    # Training loop
    best_val_rho = -1.0
    patience_ctr = 0
    global_step = 0
    epoch_times: List[float] = []
    val_rho_key = "spearman_raw" if raw_val_targets is not None else "spearman"

    for epoch in range(args.epochs):
        t0 = time.time()

        # Phase 2: unfreeze encoder
        if encoder_frozen and epoch >= args.unfreeze_epoch:
            print(f"\n* Epoch {epoch + 1}: Unfreezing encoder")
            unfreeze_encoder(model)
            encoder_frozen = False
            optimizer = build_discriminative_optimizer(
                model, args.head, args.lr, args.weight_decay,
                lr_decay=args.lr_decay_factor, encoder_lr_scale=args.encoder_lr_scale,
            )
            scheduler = CosineWarmupScheduler(
                optimizer, warmup_steps=0, total_steps=total_steps - global_step,
                min_lr_ratio=args.min_lr / args.lr if args.min_lr > 0 else 0.01,
            )
            if ema is not None:
                ema = EMAModel(model, decay=args.ema_decay)
            print(f"  Trainable (Phase 2): {fmt_params(count_params(model))}")
            patience_ctr = 0

        # Train one epoch
        model.train()
        optimizer.zero_grad(set_to_none=True)
        ep_loss = 0.0
        n_batches = 0
        grad_norm = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}", leave=True)
        for step, batch in enumerate(pbar, 1):
            tokens = batch["tokens"].to(device, non_blocking=True)
            targets = batch["label"].to(device, non_blocking=True)

            ctx = torch.autocast(device_type="cuda", dtype=amp_dtype) if use_amp else nullcontext()
            with ctx:
                if args.head == "dual" and "interaction_features" in batch:
                    inter = batch["interaction_features"].to(device, non_blocking=True)
                    preds, _ = model(tokens, inter)
                else:
                    preds, _ = model(tokens)
                loss, bm = criterion(preds, targets)
                loss_scaled = loss / grad_accum

            if scaler.is_enabled():
                scaler.scale(loss_scaled).backward()
            else:
                loss_scaled.backward()

            should_step = (step % grad_accum == 0) or (step == len(train_loader))
            if should_step:
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip).item()
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1
                if ema is not None:
                    ema.update(model)

            ep_loss += bm["loss"]
            n_batches += 1

            if step % args.log_every == 0 or step == len(train_loader):
                lr_now = optimizer.param_groups[-1]["lr"]
                pbar.set_postfix(loss=f"{bm['loss']:.4f}", ccc=f"{1-bm['ccc_loss']:.3f}", lr=f"{lr_now:.1e}")
                if args.wandb and HAS_WANDB:
                    wandb.log({
                        "train/loss": bm["loss"], "train/mse": bm["mse"],
                        "train/ccc_loss": bm["ccc_loss"], "train/lr": lr_now,
                        "train/grad_norm": grad_norm,
                    }, step=global_step)

        avg_loss = ep_loss / max(n_batches, 1)
        epoch_time = time.time() - t0
        epoch_times.append(epoch_time)

        # Validation
        if (epoch + 1) % args.val_every == 0:
            if ema is not None:
                ema.apply(model)

            vm = evaluate(model, val_loader, device, use_amp, amp_dtype,
                          head_type=args.head, raw_targets=raw_val_targets,
                          source_labels=val_source_labels)

            if ema is not None:
                ema.restore(model)

            val_rho = vm.get(val_rho_key, vm["spearman"])
            phase = "frozen" if encoder_frozen else "unfrozen"
            rho_label = "rho_raw" if val_rho_key == "spearman_raw" else "rho"

            print(
                f"  [{phase}] loss={avg_loss:.4f}  "
                f"val_{rho_label}={val_rho:.4f}  val_r={vm['pearson']:.4f}  "
                f"val_mae={vm['mae']:.4f}  time={epoch_time:.1f}s"
            )

            if args.wandb and HAS_WANDB:
                wl = {"val/spearman": vm["spearman"], "val/pearson": vm["pearson"],
                      "val/mae": vm["mae"], "val/rmse": vm["rmse"], "epoch": epoch + 1}
                if "spearman_raw" in vm:
                    wl["val/spearman_raw"] = vm["spearman_raw"]
                wandb.log(wl, step=global_step)

            if val_rho > best_val_rho:
                best_val_rho = val_rho
                patience_ctr = 0
                ckpt: Dict[str, Any] = {
                    "epoch": epoch, "global_step": global_step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_metrics": {k: v for k, v in vm.items() if k not in ("predictions", "targets")},
                    "args": vars(args), "format": fmt.name,
                    "label_mean": train_ds.label_mean, "label_std": train_ds.label_std,
                }
                if ema is not None:
                    ckpt["ema_state_dict"] = ema.state_dict()
                ckpt_path = output_dir / "best_model.pt"
                torch.save(ckpt, ckpt_path)
                print(f"  * New best {rho_label}={best_val_rho:.4f} -> {ckpt_path}")
            else:
                patience_ctr += 1
                print(f"  No improvement ({patience_ctr}/{args.patience})")

            if args.patience > 0 and patience_ctr >= args.patience:
                if encoder_frozen and args.unfreeze_epoch > 0:
                    print("  Early stopping deferred -- encoder not yet unfrozen")
                    patience_ctr = 0
                else:
                    print(f"\n  Early stopping at epoch {epoch + 1}")
                    break

    # ---- Final test evaluation ------------------------------------------------
    print(f"\n{'=' * 60}")
    print("FINAL TEST EVALUATION")
    print(f"{'=' * 60}")

    best_path = output_dir / "best_model.pt"
    if best_path.exists():
        best_ckpt = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(best_ckpt["model_state_dict"])
        if ema is not None and "ema_state_dict" in best_ckpt:
            ema.load_state_dict(best_ckpt["ema_state_dict"])
            ema.apply(model)
            print(f"Loaded EMA weights from epoch {best_ckpt['epoch'] + 1}")
        else:
            print(f"Loaded best model from epoch {best_ckpt['epoch'] + 1}")
    else:
        print("Warning: no checkpoint -- using final model")
        if ema is not None:
            ema.apply(model)

    tm = evaluate(model, test_loader, device, use_amp, amp_dtype,
                  head_type=args.head, raw_targets=raw_test_targets,
                  source_labels=test_source_labels)

    test_rho_key = "spearman_raw" if "spearman_raw" in tm else "spearman"
    test_rho = tm[test_rho_key]

    ci_tgt = raw_test_targets if (test_rho_key == "spearman_raw" and raw_test_targets is not None) else tm["targets"]
    rho_pt, rho_lo, rho_hi = bootstrap_spearman_ci(tm["predictions"], ci_tgt,
                                                    n_boot=args.bootstrap_n, seed=args.seed)

    print(f"\n  Spearman rho (z-scored) : {tm['spearman']:.4f}")
    if "spearman_raw" in tm:
        print(f"  Spearman rho (raw)     : {tm['spearman_raw']:.4f}  *primary*")
    print(f"  95% Bootstrap CI       : [{rho_lo:.4f}, {rho_hi:.4f}]  (n={args.bootstrap_n})")
    print(f"  Pearson r              : {tm['pearson']:.4f}")
    print(f"  MAE / RMSE             : {tm['mae']:.4f} / {tm['rmse']:.4f}")
    print(f"  N test                 : {tm['n_samples']}")

    if "per_source" in tm:
        print(f"\n  Per-source breakdown:")
        for src, m in sorted(tm["per_source"].items()):
            print(f"    {src:20s}  n={m['n']:5d}  rho={m['spearman']:.4f}  r={m['pearson']:.4f}")

    # Save predictions
    pred_dict: Dict[str, Any] = {"predicted": tm["predictions"], "actual": tm["targets"]}
    if raw_test_targets is not None:
        pred_dict["actual_raw"] = raw_test_targets
    if test_source_labels is not None:
        pred_dict["source"] = test_source_labels
    pd.DataFrame(pred_dict).to_csv(output_dir / "test_predictions.csv", index=False)

    # Save metrics JSON
    metrics: Dict[str, Any] = {k: v for k, v in tm.items() if k not in ("predictions", "targets")}
    metrics.update({
        "bootstrap_ci_95": {"lower": rho_lo, "upper": rho_hi, "n": args.bootstrap_n},
        "best_val_spearman": best_val_rho,
        "total_epochs": epoch + 1,
        "mean_epoch_time_s": float(np.mean(epoch_times)),
        "total_time_s": float(np.sum(epoch_times)),
        "head": args.head, "format": fmt.name, "guide_len": args.guide_len,
        "loss_weights": {"mse": args.mse_weight, "ccc": args.ccc_weight, "huber": args.huber_weight},
        "ema": args.ema, "environment": env_info,
    })
    if train_ds.label_mean is not None:
        metrics["zscore"] = {"mean": train_ds.label_mean, "std": train_ds.label_std}
    with open(output_dir / "test_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    print(f"\n  Saved: {output_dir}/test_predictions.csv")
    print(f"  Saved: {output_dir}/test_metrics.json")

    if args.wandb and HAS_WANDB:
        wl = {"test/spearman": tm["spearman"], "test/pearson": tm["pearson"],
              "test/mae": tm["mae"], "test/rho_ci_lo": rho_lo, "test/rho_ci_hi": rho_hi}
        if "spearman_raw" in tm:
            wl["test/spearman_raw"] = tm["spearman_raw"]
        wandb.log(wl)
        wandb.finish()

    rho_l = "rho_raw" if "spearman_raw" in tm else "rho"
    print(f"\n{'=' * 60}")
    print(f"Done -- val {rho_l}={best_val_rho:.4f}, test {rho_l}={test_rho:.4f} [{rho_lo:.4f}, {rho_hi:.4f}]")
    print(f"{'=' * 60}")


# --- CLI ----------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fine-tune Cas12a activity predictor (DNA-JEPA)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    g = p.add_argument_group("paths")
    g.add_argument("--data", default="data/processed/cas12a_efficiency.csv")
    g.add_argument("--pretrained-path", default="checkpoints/pretrain/checkpoint_epoch100.pt")
    g.add_argument("--output-dir", default="checkpoints/finetune")

    g = p.add_argument_group("architecture")
    g.add_argument("--head", choices=["mlp", "dual", "legacy"], default="mlp")
    g.add_argument("--embed-dim", type=int, default=384)
    g.add_argument("--num-layers", type=int, default=6)
    g.add_argument("--num-heads", type=int, default=6)
    g.add_argument("--ff-dim", type=int, default=1024)
    g.add_argument("--head-dropout", type=float, default=0.2)
    g.add_argument("--guide-len", type=int, default=23)
    g.add_argument("--guide-len-auto", action="store_true", default=True)
    g.add_argument("--no-guide-len-auto", dest="guide_len_auto", action="store_false")
    g.add_argument("--n-interaction-channels", type=int, default=10)
    g.add_argument("--max-tokens", type=int, default=512)

    g = p.add_argument_group("training")
    g.add_argument("--epochs", type=int, default=100)
    g.add_argument("--batch-size", type=int, default=64)
    g.add_argument("--lr", type=float, default=3e-4)
    g.add_argument("--min-lr", type=float, default=1e-6)
    g.add_argument("--warmup-epochs", type=int, default=5)
    g.add_argument("--weight-decay", type=float, default=1e-2)
    g.add_argument("--grad-clip", type=float, default=1.0)
    g.add_argument("--grad-accum-steps", type=int, default=1)

    g = p.add_argument_group("loss")
    g.add_argument("--mse-weight", type=float, default=0.5)
    g.add_argument("--ccc-weight", type=float, default=0.3)
    g.add_argument("--huber-weight", type=float, default=0.2)
    g.add_argument("--huber-delta", type=float, default=1.0)

    g = p.add_argument_group("discriminative LR")
    g.add_argument("--unfreeze-epoch", type=int, default=10)
    g.add_argument("--lr-decay-factor", type=float, default=0.8)
    g.add_argument("--encoder-lr-scale", type=float, default=0.1)

    g = p.add_argument_group("regularisation")
    g.add_argument("--patience", type=int, default=15)
    g.add_argument("--ema", action="store_true", default=False)
    g.add_argument("--ema-decay", type=float, default=0.999)

    g = p.add_argument_group("data")
    g.add_argument("--val-fraction", type=float, default=0.15)
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--deterministic", action="store_true", default=False)

    g = p.add_argument_group("performance")
    g.add_argument("--precision", choices=["auto", "bf16", "fp16", "fp32"], default="auto")
    g.add_argument("--num-workers", type=int, default=0)
    g.add_argument("--prefetch-factor", type=int, default=2)

    g = p.add_argument_group("evaluation")
    g.add_argument("--bootstrap-n", type=int, default=1000)

    g = p.add_argument_group("logging")
    g.add_argument("--log-every", type=int, default=20)
    g.add_argument("--val-every", type=int, default=1)
    g.add_argument("--wandb", action="store_true")
    g.add_argument("--wandb-project", default="dna-jepa-cas12a")
    g.add_argument("--wandb-name", default=None)

    return p.parse_args()


if __name__ == "__main__":
    finetune(parse_args())
