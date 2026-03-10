"""
Pretraining script for DNA-Bacteria-JEPA (v4 — SIGReg + W&B).

v4 changes from v3
-------------------
- **SIGReg** replaces VICReg + LDReg + SupCon (1 hyperparameter vs 8+)
  (Balestriero & LeCun, LeJEPA, Nov 2025)
- Loss = MSE prediction + λ·SIGReg + RC consistency + GC adversary
- Optimised for 1K-genome scale pretraining

Retained from v3
-----------------
- Multi-block masking (I-JEPA, 4 target blocks)
- Transformer predictor with mask tokens (384-D bottleneck)
- Curriculum masking (easy → hard)
- Cosine LR + cosine EMA + gradient accumulation
- Weights & Biases live tracking

Usage
-----
Fresh start::

    python scripts/01_pretrain_jepa.py \\
        --data-path data/processed/pretrain_sequences_expanded.csv \\
        --epochs 100 --batch-size 256 --lr 3e-4 \\
        --precision auto --save-every 10

Resume::

    python scripts/01_pretrain_jepa.py \\
        --resume checkpoints/pretrain/checkpoint_epoch50.pt

"""

from __future__ import annotations

import argparse
import copy
import math
import random
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F
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
    print("Warning: wandb not installed. Run `pip install wandb` for tracking.")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.cas12a.dataset import BacterialGenomeDataset
from src.cas12a.encoder import SparseTransformerEncoder
from src.cas12a.jepa_pretrain import (
    Cas12aJEPA,
    GCAdversary,
    JEPAConfig,
    MaskingConfig,
    SIGRegConfig,
    compute_gc_content,
    compute_prediction_loss,
    compute_rankme,
    compute_sigreg_loss,
    curriculum_masking_params,
    gc_correlation,
    reverse_complement_tokens,
)
from src.cas12a.tokenizer import Cas12aTokenizer, TokenizerConfig


# ═══════════════════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════════════════

def resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_amp_dtype(precision: str, device: torch.device):
    if device.type != "cuda" or precision == "fp32":
        return None
    if precision == "auto":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if precision == "bf16":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if precision == "fp16":
        return torch.float16
    raise ValueError(f"Unsupported precision: {precision}")


def cosine_lr(optimizer, step, total_steps, warmup_steps, peak_lr, min_lr=1e-6):
    if step < warmup_steps:
        lr = peak_lr * (step / max(warmup_steps, 1))
    else:
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        lr = min_lr + 0.5 * (peak_lr - min_lr) * (1 + math.cos(math.pi * progress))
    for pg in optimizer.param_groups:
        pg["lr"] = lr
    return lr


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_time(seconds):
    h, m, s = int(seconds // 3600), int((seconds % 3600) // 60), int(seconds % 60)
    return f"{h}:{m:02d}:{s:02d}"


def build_complement_map(tokenizer):
    base_pairs = [('A', 'T'), ('C', 'G')]
    comp_map = {}
    for b1, b2 in base_pairs:
        id1, id2 = None, None
        if hasattr(tokenizer, 'base_to_id'):
            id1, id2 = tokenizer.base_to_id.get(b1), tokenizer.base_to_id.get(b2)
        if id1 is None and hasattr(tokenizer, 'char_to_id'):
            id1, id2 = tokenizer.char_to_id.get(b1), tokenizer.char_to_id.get(b2)
        if id1 is None and hasattr(tokenizer, 'encode'):
            try:
                enc1, enc2 = tokenizer.encode(b1), tokenizer.encode(b2)
                id1 = enc1[0] if isinstance(enc1, list) else enc1
                id2 = enc2[0] if isinstance(enc2, list) else enc2
            except Exception:
                pass
        if id1 is None:
            id1, id2 = {'A': 1, 'C': 2, 'G': 3, 'T': 4}.get(b1), {'A': 1, 'C': 2, 'G': 3, 'T': 4}.get(b2)
        if id1 is not None and id2 is not None:
            comp_map[id1] = id2
            comp_map[id2] = id1
    return comp_map


def build_gc_token_ids(tokenizer):
    gc_ids = set()
    for attr in ['base_to_id', 'char_to_id']:
        if hasattr(tokenizer, attr):
            mapping = getattr(tokenizer, attr)
            for base in ['G', 'C']:
                tid = mapping.get(base)
                if tid is not None:
                    gc_ids.add(tid)
            if gc_ids:
                return gc_ids
    return {2, 3}  # fallback


@torch.no_grad()
def evaluate_epoch(jepa_model, dataloader, tokenizer, device, max_batches=30, gc_token_ids=None):
    jepa_model.eval()
    all_embeddings, all_tokens = [], []
    for i, batch in enumerate(dataloader):
        if i >= max_batches:
            break
        tokens = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)
        attention_mask = tokenizer.get_attention_mask(tokens)
        emb = jepa_model.encode(tokens, attention_mask=attention_mask)
        all_embeddings.append(emb.cpu())
        all_tokens.append(tokens.cpu())

    embeddings = torch.cat(all_embeddings, dim=0)
    tokens = torch.cat(all_tokens, dim=0)
    rankme = compute_rankme(embeddings)
    gc_abs, gc_raw = gc_correlation(tokens, embeddings, pad_token_id=tokenizer.pad_id, gc_token_ids=gc_token_ids)
    pred_std = embeddings.std(dim=0).mean().item()

    # SIGReg gaussianity score (on eval embeddings)
    sigreg_score = jepa_model.sigreg.gaussianity_score(embeddings.to(device))

    return {
        "rankme": rankme,
        "gc_abs_r": gc_abs,
        "gc_raw_r": gc_raw,
        "pred_std": pred_std,
        "sigreg_eval": sigreg_score,
    }


class GenomeLabelDataset(torch.utils.data.Dataset):
    """Wrapper adding genome labels (for logging, not for loss in v4)."""
    def __init__(self, base_dataset):
        self.base = base_dataset
        if hasattr(self.base, 'df') and 'genome' in self.base.df.columns:
            genomes = self.base.df['genome'].values
            unique_genomes = sorted(set(genomes))
            self.genome_to_id = {g: i for i, g in enumerate(unique_genomes)}
            self.genome_ids = [self.genome_to_id[g] for g in genomes]
            self.n_genomes = len(unique_genomes)
            print(f"  Genome labels: {self.n_genomes} unique genomes")
        else:
            self.genome_ids = [0] * len(self.base)
            self.n_genomes = 0
            print("  No 'genome' column — genome-level metrics disabled")

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        return self.base[idx], self.genome_ids[idx]

    @property
    def df(self):
        return self.base.df

    @df.setter
    def df(self, value):
        self.base.df = value


# ═══════════════════════════════════════════════════════════════════════════
# Main training loop
# ═══════════════════════════════════════════════════════════════════════════

def pretrain(args):
    set_seed(args.seed)
    data_path = resolve_path(args.data_path)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not data_path.exists():
        print(f"Data not found: {data_path}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    # ── Dataset ──
    tokenizer = Cas12aTokenizer(TokenizerConfig())
    print(f"Tokenizer: vocab_size={tokenizer.vocab_size}")

    base_dataset = BacterialGenomeDataset(str(data_path), tokenizer)
    if 0 < args.max_samples < len(base_dataset):
        base_dataset.df = base_dataset.df.sample(n=args.max_samples, random_state=args.seed).reset_index(drop=True)
        print(f"Dataset: {len(base_dataset):,} sequences (sampled)")
    else:
        print(f"Dataset: {len(base_dataset):,} sequences")

    dataset = GenomeLabelDataset(base_dataset)
    complement_map = build_complement_map(tokenizer)
    gc_token_ids = build_gc_token_ids(tokenizer)
    print(f"  Complement map: {complement_map}")
    print(f"  GC token IDs: {gc_token_ids}")

    dl_kwargs = {
        "batch_size": args.batch_size, "shuffle": True,
        "num_workers": args.num_workers, "pin_memory": device.type == "cuda",
        "drop_last": True,
    }
    if args.num_workers > 0:
        dl_kwargs["persistent_workers"] = True
        dl_kwargs["prefetch_factor"] = args.prefetch_factor
    dataloader = DataLoader(dataset, **dl_kwargs)
    steps_per_epoch = len(dataloader)

    # ── Model ──
    encoder = SparseTransformerEncoder(
        vocab_size=tokenizer.vocab_size,
        embed_dim=args.embed_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        ff_dim=args.ff_dim,
    )

    max_seq_len = args.max_seq_len
    if hasattr(base_dataset, 'df') and 'sequence' in base_dataset.df.columns:
        sample_len = len(base_dataset.df['sequence'].iloc[0]) + 10
        max_seq_len = max(max_seq_len, sample_len)
        print(f"  Max sequence length (from data): {max_seq_len}")

    jepa_config = JEPAConfig(
        predictor_dim=args.predictor_dim,
        predictor_depth=args.predictor_depth,
        predictor_num_heads=args.predictor_num_heads,
        max_seq_len=max_seq_len,
        ema_decay_start=args.ema_decay_start,
        ema_decay_end=args.ema_decay_end,
        masking=MaskingConfig(
            mask_ratio_start=args.mask_ratio_start,
            mask_ratio_end=args.mask_ratio_end,
            num_target_blocks=args.num_target_blocks,
            min_block_len_start=args.min_block_len_start,
            min_block_len_end=args.min_block_len_end,
            context_ratio_floor=args.context_ratio_floor,
        ),
        sigreg=SIGRegConfig(
            weight=args.sigreg_weight,
            num_sketches=args.sigreg_num_sketches,
        ),
    )

    jepa_model = Cas12aJEPA(encoder, config=jepa_config)
    jepa_model.to(device)

    n_enc = count_parameters(jepa_model.context_encoder)
    n_pred = count_parameters(jepa_model.predictor)
    print(f"\nEncoder: {n_enc / 1e6:.1f}M params | Predictor: {n_pred / 1e6:.1f}M params")
    print(f"  Predictor: {args.predictor_depth}L × {args.predictor_dim}D × {args.predictor_num_heads}H")

    # ── GC Adversary ──
    gc_adversary = GCAdversary(embed_dim=args.embed_dim, hidden_dim=args.gc_adv_hidden_dim)
    gc_adversary.to(device)
    print(f"GC Adversary: {count_parameters(gc_adversary) / 1e3:.1f}K params")

    # ── Optimizer ──
    trainable_params = [
        {"params": jepa_model.context_encoder.parameters(), "lr": args.lr},
        {"params": jepa_model.predictor.parameters(), "lr": args.lr},
        {"params": gc_adversary.parameters(), "lr": args.lr},
    ]
    optimizer = torch.optim.AdamW(
        trainable_params, lr=args.lr,
        weight_decay=args.weight_decay, betas=(0.9, 0.999),
    )

    # ── AMP ──
    amp_dtype = select_amp_dtype(args.precision, device)
    use_amp = amp_dtype is not None
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and amp_dtype == torch.float16))
    grad_accum = max(1, args.grad_accum_steps)

    total_steps = args.epochs * steps_per_epoch
    warmup_steps = args.warmup_epochs * steps_per_epoch

    # ── Resume ──
    start_epoch, global_step, best_loss = 0, 0, float("inf")

    if args.resume:
        ckpt_path = resolve_path(args.resume)
        if not ckpt_path.exists():
            print(f"Checkpoint not found: {ckpt_path}")
            return
        print(f"\n── Resuming from {ckpt_path} ──")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        jepa_model.context_encoder.load_state_dict(ckpt["encoder_state_dict"])
        print("  Restored: context encoder")
        if "target_encoder_state_dict" in ckpt:
            jepa_model.target_encoder.load_state_dict(ckpt["target_encoder_state_dict"])
            print("  Restored: target encoder")
        else:
            jepa_model.target_encoder = copy.deepcopy(jepa_model.context_encoder)
            for p in jepa_model.target_encoder.parameters():
                p.requires_grad = False
            print("  Target encoder: re-copied")
        if "predictor_state_dict" in ckpt:
            try:
                jepa_model.predictor.load_state_dict(ckpt["predictor_state_dict"])
                print("  Restored: predictor")
            except (RuntimeError, KeyError) as e:
                print(f"  Predictor: re-initialised ({e})")
        if "gc_adversary_state_dict" in ckpt:
            gc_adversary.load_state_dict(ckpt["gc_adversary_state_dict"])
            print("  Restored: GC adversary")
        if "optimizer_state_dict" in ckpt:
            try:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                print("  Restored: optimizer")
            except (ValueError, KeyError) as e:
                print(f"  Optimizer: re-initialised ({e})")
        if "scaler_state_dict" in ckpt and ckpt["scaler_state_dict"] is not None and scaler.is_enabled():
            try:
                scaler.load_state_dict(ckpt["scaler_state_dict"])
            except Exception:
                pass
        start_epoch = ckpt.get("epoch", 0) + 1
        global_step = start_epoch * steps_per_epoch
        best_loss = ckpt.get("loss", float("inf"))
        print(f"  Resuming at epoch {start_epoch + 1}/{args.epochs}, prev loss: {best_loss:.4f}\n")

    # ── Print config ──
    eff_batch = args.batch_size * grad_accum
    prec_label = str(amp_dtype).replace("torch.", "") if use_amp else "fp32"

    print(f"\n{'─' * 70}")
    print(f"  DNA-Bacteria-JEPA v4 — SIGReg (LeJEPA)")
    print(f"{'─' * 70}")
    print(f"  Batch: {args.batch_size} × accum={grad_accum} = eff={eff_batch}, precision={prec_label}")
    print(f"  Schedule: {args.epochs} epochs, warmup={args.warmup_epochs}, lr={args.lr:.1e}→{args.min_lr:.1e}")
    print(f"  EMA: τ={args.ema_decay_start}→{args.ema_decay_end} (cosine)")
    print(f"  SIGReg: λ={args.sigreg_weight}, K={args.sigreg_num_sketches} sketches")
    print(f"  Seq-level prediction: weight={args.seq_pred_weight}")
    print(f"  Multi-block masking: {args.num_target_blocks} blocks")
    print(f"  Curriculum: ratio={args.mask_ratio_start:.2f}→{args.mask_ratio_end:.2f}, "
          f"block_len={args.min_block_len_start}→{args.min_block_len_end}")
    print(f"  Predictor: {args.predictor_depth}L × {args.predictor_dim}D (I-JEPA transformer)")
    print(f"  Auxiliary: RC={args.rc_weight}, GC_adv={args.gc_adv_weight}")

    # ── W&B ──
    use_wandb = HAS_WANDB and not args.no_wandb
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config={
                "version": "v4-sigreg",
                "encoder_params": n_enc, "predictor_params": n_pred,
                "embed_dim": args.embed_dim, "num_layers": args.num_layers,
                "num_heads": args.num_heads, "ff_dim": args.ff_dim,
                "predictor_dim": args.predictor_dim, "predictor_depth": args.predictor_depth,
                "epochs": args.epochs, "batch_size": args.batch_size,
                "effective_batch_size": eff_batch, "lr": args.lr,
                "min_lr": args.min_lr, "warmup_epochs": args.warmup_epochs,
                "weight_decay": args.weight_decay, "precision": prec_label,
                "sigreg_weight": args.sigreg_weight,
                "sigreg_num_sketches": args.sigreg_num_sketches,
                "seq_pred_weight": args.seq_pred_weight,
                "rc_weight": args.rc_weight,
                "gc_adv_weight": args.gc_adv_weight,
                "num_target_blocks": args.num_target_blocks,
                "mask_ratio_start": args.mask_ratio_start,
                "mask_ratio_end": args.mask_ratio_end,
                "min_block_len_start": args.min_block_len_start,
                "min_block_len_end": args.min_block_len_end,
                "ema_decay_start": args.ema_decay_start,
                "ema_decay_end": args.ema_decay_end,
                "dataset_size": len(dataset),
                "n_genomes": dataset.n_genomes,
                "max_seq_len": max_seq_len,
                "steps_per_epoch": steps_per_epoch,
                "total_steps": total_steps,
            },
            tags=["v4", "SIGReg", "JEPA", "DNA", "pretrain"],
            notes=f"DNA-Bacteria-JEPA v4 (SIGReg): {n_enc/1e6:.1f}M enc, {len(dataset):,} seqs, {args.epochs} ep",
            resume="allow" if args.resume else None,
        )
        wandb.watch(jepa_model.context_encoder, log="gradients", log_freq=100)
        print(f"  W&B: {wandb.run.url}")
    else:
        print(f"  W&B: disabled")

    print(f"\n{'═' * 70}")
    print(f"  Starting pretraining: {args.epochs - start_epoch} epochs remaining")
    print(f"{'═' * 70}\n")

    # ══════════════════════════════════════════════════════════════════
    # Training loop
    # ══════════════════════════════════════════════════════════════════
    training_start = time.time()

    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()
        jepa_model.train()
        gc_adversary.train()

        progress = epoch / max(args.epochs - 1, 1)
        current_ema = jepa_model.set_ema_decay(progress)
        cur_mask_ratio, cur_min_block_len = curriculum_masking_params(
            epoch, args.epochs, jepa_config.masking,
        )
        lambda_gc = GCAdversary.ganin_lambda(epoch, args.epochs) if args.gc_adv_weight > 0 else 0.0

        # Accumulators
        epoch_losses = []
        epoch_components = {
            "pred_mse": [], "seq_pred": [], "sigreg": [],
            "rc": [], "gc_adv": [], "cos_sim": [],
        }
        optimizer.zero_grad(set_to_none=True)

        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{args.epochs}", leave=True, ncols=140)

        for step, batch in enumerate(pbar, start=1):
            global_step += 1
            current_lr = cosine_lr(optimizer, global_step, total_steps, warmup_steps, args.lr, args.min_lr)

            tokens = batch[0] if isinstance(batch, (list, tuple)) else batch
            tokens = tokens.to(device, non_blocking=True)
            attention_mask = tokenizer.get_attention_mask(tokens)

            autocast_ctx = torch.autocast("cuda", dtype=amp_dtype) if use_amp else nullcontext()

            with autocast_ctx:
                # ── Forward pass ──
                pred_emb, target_emb, info = jepa_model(
                    tokens, attention_mask=attention_mask,
                    mask_ratio=cur_mask_ratio, min_block_len=cur_min_block_len,
                    pad_token_id=tokenizer.pad_id,
                )

                # ── Loss 1: Token-level prediction (MSE) ──
                pred_loss, pred_metrics = compute_prediction_loss(pred_emb, target_emb)
                total_loss = pred_loss

                # ── Loss 2: Sequence-level prediction ──
                seq_pred_val = 0.0
                if args.seq_pred_weight > 0:
                    seq_pred_loss = F.mse_loss(info["context_pooled"], info["target_pooled"])
                    total_loss = total_loss + args.seq_pred_weight * seq_pred_loss
                    seq_pred_val = seq_pred_loss.item()

                # ── Loss 3: SIGReg (replaces VICReg + LDReg + SupCon) ──
                sigreg_val = 0.0
                if args.sigreg_weight > 0:
                    sigreg_loss, sigreg_metrics = compute_sigreg_loss(
                        info["context_pooled"], jepa_model.sigreg,
                    )
                    total_loss = total_loss + args.sigreg_weight * sigreg_loss
                    sigreg_val = sigreg_loss.item()

                # ── Loss 4: RC consistency ──
                rc_val = 0.0
                if args.rc_weight > 0 and len(complement_map) >= 4:
                    rc_tokens = reverse_complement_tokens(tokens, complement_map, pad_token_id=tokenizer.pad_id)
                    rc_attn = tokenizer.get_attention_mask(rc_tokens)
                    rc_pooled, _ = jepa_model.context_encoder(rc_tokens, attention_mask=rc_attn)
                    rc_loss = F.mse_loss(info["context_pooled"], rc_pooled)
                    total_loss = total_loss + args.rc_weight * rc_loss
                    rc_val = rc_loss.item()

                # ── Loss 5: GC adversary ──
                gc_adv_val = 0.0
                if args.gc_adv_weight > 0:
                    gc_targets = compute_gc_content(tokens, pad_token_id=tokenizer.pad_id, gc_token_ids=gc_token_ids)
                    gc_pred = gc_adversary(info["context_pooled"], lambda_=lambda_gc)
                    gc_adv_loss = F.mse_loss(gc_pred, gc_targets)
                    total_loss = total_loss + args.gc_adv_weight * gc_adv_loss
                    gc_adv_val = gc_adv_loss.item()

                scaled_loss = total_loss / grad_accum

            # ── Backward ──
            if scaler.is_enabled():
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

            should_step = (step % grad_accum == 0) or (step == steps_per_epoch)
            if should_step:
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(jepa_model.context_encoder.parameters(), args.grad_clip)
                torch.nn.utils.clip_grad_norm_(jepa_model.predictor.parameters(), args.grad_clip)
                torch.nn.utils.clip_grad_norm_(gc_adversary.parameters(), args.grad_clip)
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                jepa_model.update_ema()

            # ── Accumulate ──
            epoch_losses.append(total_loss.item())
            epoch_components["pred_mse"].append(pred_metrics["pred_mse"])
            epoch_components["seq_pred"].append(seq_pred_val)
            epoch_components["sigreg"].append(sigreg_val)
            epoch_components["rc"].append(rc_val)
            epoch_components["gc_adv"].append(gc_adv_val)
            epoch_components["cos_sim"].append(pred_metrics["pred_cos_sim"])

            # ── Per-step logging ──
            if use_wandb and ((step % args.log_every == 0) or step == steps_per_epoch):
                wandb.log({
                    "step/total_loss": total_loss.item(),
                    "step/pred_mse": pred_metrics["pred_mse"],
                    "step/pred_cos_sim": pred_metrics["pred_cos_sim"],
                    "step/seq_pred": seq_pred_val,
                    "step/sigreg": sigreg_val,
                    "step/rc": rc_val,
                    "step/gc_adv": gc_adv_val,
                    "step/lr": current_lr,
                }, step=global_step)

            if (step % args.log_every == 0) or step == steps_per_epoch:
                pbar.set_postfix({
                    "loss": f"{total_loss.item():.3f}",
                    "mse": f"{pred_metrics['pred_mse']:.3f}",
                    "cos": f"{pred_metrics['pred_cos_sim']:.3f}",
                    "sig": f"{sigreg_val:.4f}",
                    "rc": f"{rc_val:.3f}",
                    "gc": f"{gc_adv_val:.4f}",
                    "mr": f"{cur_mask_ratio:.2f}",
                    "lr": f"{current_lr:.1e}",
                })

        # ── Epoch summary ──
        epoch_time = time.time() - epoch_start
        avg = {k: sum(v) / max(len(v), 1) for k, v in epoch_components.items()}
        avg_loss = sum(epoch_losses) / max(len(epoch_losses), 1)

        eval_metrics = evaluate_epoch(
            jepa_model, dataloader, tokenizer, device,
            max_batches=args.eval_batches, gc_token_ids=gc_token_ids,
        )

        elapsed = time.time() - training_start
        remaining_est = (elapsed / max(epoch - start_epoch + 1, 1)) * (args.epochs - epoch - 1)

        print(f"\n  Epoch {epoch + 1}/{args.epochs} ({epoch_time:.1f}s, ETA {format_time(remaining_est)})")
        print(f"  Total: {avg_loss:.4f}  pred_mse={avg['pred_mse']:.4f}  cos_sim={avg['cos_sim']:.3f}")
        print(f"  SIGReg: {avg['sigreg']:.5f} (λ={args.sigreg_weight})  "
              f"seq_pred={avg['seq_pred']:.4f}  RC={avg['rc']:.4f}  GC_adv={avg['gc_adv']:.5f}")
        print(f"  Curriculum: mask_ratio={cur_mask_ratio:.3f}  min_block_len={cur_min_block_len}")
        print(f"  RankMe: {eval_metrics['rankme']:.1f}/{args.embed_dim}  "
              f"GC |r|={eval_metrics['gc_abs_r']:.3f}  "
              f"pred_std={eval_metrics['pred_std']:.3f}  "
              f"sigreg_eval={eval_metrics['sigreg_eval']:.4f}")
        print(f"  LR={current_lr:.2e}  EMA τ={current_ema:.4f}")

        # ── W&B epoch logging ──
        if use_wandb:
            wandb.log({
                "epoch/total_loss": avg_loss,
                "epoch/pred_mse": avg["pred_mse"],
                "epoch/pred_cos_sim": avg["cos_sim"],
                "epoch/seq_pred": avg["seq_pred"],
                "epoch/sigreg": avg["sigreg"],
                "epoch/rc": avg["rc"],
                "epoch/gc_adv": avg["gc_adv"],
                "health/rankme": eval_metrics["rankme"],
                "health/rankme_ratio": eval_metrics["rankme"] / args.embed_dim,
                "health/pred_std": eval_metrics["pred_std"],
                "health/gc_abs_r": eval_metrics["gc_abs_r"],
                "health/sigreg_eval": eval_metrics["sigreg_eval"],
                "schedule/lr": current_lr,
                "schedule/ema_tau": current_ema,
                "schedule/mask_ratio": cur_mask_ratio,
                "schedule/min_block_len": cur_min_block_len,
                "schedule/lambda_gc": lambda_gc,
                "epoch": epoch + 1,
                "epoch_time_s": epoch_time,
                "eta_s": remaining_est,
            }, step=global_step)

        if avg_loss < best_loss:
            best_loss = avg_loss

        # ── Save checkpoint ──
        if (epoch + 1) % args.save_every == 0 or (epoch + 1) == args.epochs:
            ckpt_path = output_dir / f"checkpoint_epoch{epoch + 1}.pt"
            torch.save({
                "epoch": epoch, "global_step": global_step,
                "encoder_state_dict": jepa_model.context_encoder.state_dict(),
                "target_encoder_state_dict": jepa_model.target_encoder.state_dict(),
                "predictor_state_dict": jepa_model.predictor.state_dict(),
                "gc_adversary_state_dict": gc_adversary.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict() if scaler.is_enabled() else None,
                "loss": avg_loss,
                "metrics": {
                    "rankme": eval_metrics["rankme"],
                    "gc_abs_r": eval_metrics["gc_abs_r"],
                    "pred_std": eval_metrics["pred_std"],
                    "sigreg_eval": eval_metrics["sigreg_eval"],
                    "avg_pred_mse": avg["pred_mse"],
                    "avg_cos_sim": avg["cos_sim"],
                    "avg_rc": avg["rc"],
                    "avg_gc_adv": avg["gc_adv"],
                    "cur_mask_ratio": cur_mask_ratio,
                    "cur_min_block_len": cur_min_block_len,
                },
                "config": {
                    "version": "v4",
                    "regularizer": "sigreg",
                    "predictor_type": "transformer_mask_tokens",
                    "multi_block_masking": True,
                    "curriculum_masking": True,
                },
                "args": vars(args),
            }, ckpt_path)
            print(f"  Saved: {ckpt_path}")

            if use_wandb and args.wandb_save_checkpoints:
                artifact = wandb.Artifact(
                    f"jepa-v4-ep{epoch+1}", type="model",
                    metadata={"epoch": epoch+1, "loss": avg_loss},
                )
                artifact.add_file(str(ckpt_path))
                wandb.log_artifact(artifact)

    # ── Finish ──
    total_time = time.time() - training_start
    print(f"\n{'═' * 70}")
    print(f"  Pretraining (v4 — SIGReg) complete in {format_time(total_time)}")
    print(f"  Best loss: {best_loss:.4f}")
    print(f"  Final RankMe: {eval_metrics['rankme']:.1f}/{args.embed_dim}")
    print(f"  Final GC |r|: {eval_metrics['gc_abs_r']:.3f}")
    print(f"  Final SIGReg: {eval_metrics['sigreg_eval']:.4f}")
    print(f"{'═' * 70}")

    if use_wandb:
        wandb.summary["best_loss"] = best_loss
        wandb.summary["final_rankme"] = eval_metrics["rankme"]
        wandb.summary["final_gc_abs_r"] = eval_metrics["gc_abs_r"]
        wandb.summary["final_sigreg_eval"] = eval_metrics["sigreg_eval"]
        wandb.summary["final_pred_std"] = eval_metrics["pred_std"]
        wandb.summary["total_training_time_h"] = total_time / 3600
        wandb.finish()
        print("  W&B: finished and synced.")


def build_parser():
    p = argparse.ArgumentParser(
        description="DNA-Bacteria-JEPA v4 Pretraining (SIGReg + W&B)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    g = p.add_argument_group("Data & IO")
    g.add_argument("--data-path", default="data/processed/pretrain_sequences.csv")
    g.add_argument("--output-dir", default="checkpoints/pretrain")
    g.add_argument("--resume", type=str, default=None)
    g.add_argument("--max-samples", type=int, default=0)
    g.add_argument("--seed", type=int, default=42)

    g = p.add_argument_group("Training")
    g.add_argument("--epochs", type=int, default=100)
    g.add_argument("--batch-size", type=int, default=256)
    g.add_argument("--lr", type=float, default=3e-4)
    g.add_argument("--min-lr", type=float, default=1e-6)
    g.add_argument("--warmup-epochs", type=int, default=10)
    g.add_argument("--weight-decay", type=float, default=0.05)
    g.add_argument("--grad-clip", type=float, default=1.0)
    g.add_argument("--grad-accum-steps", type=int, default=2)
    g.add_argument("--precision", choices=["auto", "bf16", "fp16", "fp32"], default="auto")

    g = p.add_argument_group("Encoder")
    g.add_argument("--embed-dim", type=int, default=384)
    g.add_argument("--num-layers", type=int, default=6)
    g.add_argument("--num-heads", type=int, default=6)
    g.add_argument("--ff-dim", type=int, default=1024)

    g = p.add_argument_group("JEPA Predictor")
    g.add_argument("--predictor-dim", type=int, default=384)
    g.add_argument("--predictor-depth", type=int, default=4)
    g.add_argument("--predictor-num-heads", type=int, default=6)
    g.add_argument("--max-seq-len", type=int, default=1024)
    g.add_argument("--ema-decay-start", type=float, default=0.996)
    g.add_argument("--ema-decay-end", type=float, default=1.0)

    g = p.add_argument_group("Multi-Block Masking + Curriculum")
    g.add_argument("--num-target-blocks", type=int, default=4)
    g.add_argument("--mask-ratio-start", type=float, default=0.15)
    g.add_argument("--mask-ratio-end", type=float, default=0.50)
    g.add_argument("--min-block-len-start", type=int, default=5)
    g.add_argument("--min-block-len-end", type=int, default=20)
    g.add_argument("--context-ratio-floor", type=float, default=0.30)

    g = p.add_argument_group("SIGReg (LeJEPA)")
    g.add_argument("--sigreg-weight", type=float, default=1.0,
                    help="λ for SIGReg regularization (single hyperparameter)")
    g.add_argument("--sigreg-num-sketches", type=int, default=1024,
                    help="K random projection directions for Epps-Pulley test")

    g = p.add_argument_group("Sequence-Level Prediction")
    g.add_argument("--seq-pred-weight", type=float, default=0.5,
                    help="Weight for sequence-level MSE (context_pooled vs target_pooled)")

    g = p.add_argument_group("Reverse Complement")
    g.add_argument("--rc-weight", type=float, default=0.1)

    g = p.add_argument_group("GC Adversary")
    g.add_argument("--gc-adv-weight", type=float, default=1.0)
    g.add_argument("--gc-adv-hidden-dim", type=int, default=64)

    g = p.add_argument_group("Weights & Biases")
    g.add_argument("--no-wandb", action="store_true")
    g.add_argument("--wandb-project", type=str, default="DNA-Bacteria-JEPA")
    g.add_argument("--wandb-run-name", type=str, default=None)
    g.add_argument("--wandb-save-checkpoints", action="store_true")

    g = p.add_argument_group("Logging")
    g.add_argument("--save-every", type=int, default=10)
    g.add_argument("--log-every", type=int, default=20)
    g.add_argument("--eval-batches", type=int, default=30)

    g = p.add_argument_group("DataLoader")
    g.add_argument("--num-workers", type=int, default=4)
    g.add_argument("--prefetch-factor", type=int, default=2)

    return p


if __name__ == "__main__":
    pretrain(build_parser().parse_args())
