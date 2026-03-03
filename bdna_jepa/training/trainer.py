"""Main training loop for B-JEPA pretraining.

Features:
    - Mixed precision (bfloat16 on A100, fp16 fallback)
    - Cosine LR with linear warmup
    - Cosine weight decay schedule (0.04 -> 0.4)
    - Stop-gradient target encoder (default) or EMA
    - GradNorm loss balancing
    - RankMe monitoring
    - Checkpoint saving with optimizer state
    - W&B logging
    - torch.compile support
    - Gradient clipping

Patterns from Subliminal 1.3 training:
    - GradScaler for mixed precision
    - Checkpoint + optimizer state saving/loading
    - Validation at intervals
"""
from __future__ import annotations

import math
import os
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from bdna_jepa.config import BJEPAConfig, TrainingConfig, LossConfig
from bdna_jepa.models.jepa import BJEPA
from bdna_jepa.losses.criterion import BJEPACriterion
from bdna_jepa.data.masking import random_mask, span_mask
from bdna_jepa.utils.metrics import compute_rankme, compute_feature_std
from bdna_jepa.utils.logging import get_logger, log_metrics, log_checkpoint


logger = get_logger(__name__)


class BJEPATrainer:
    """B-JEPA pretraining trainer."""

    def __init__(
        self,
        model: BJEPA,
        criterion: BJEPACriterion,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        model_config: BJEPAConfig,
        train_config: TrainingConfig,
        device: torch.device,
    ):
        self.model = model.to(device)
        self.criterion = criterion.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.mc = model_config
        self.tc = train_config
        self.device = device

        # Optimizer
        self.optimizer = self._build_optimizer()

        # Mixed precision
        self.use_amp = train_config.mixed_precision and device.type == "cuda"
        self.amp_dtype = torch.bfloat16 if train_config.precision == "bf16" else torch.float16
        self.scaler = torch.amp.GradScaler(enabled=(self.use_amp and self.amp_dtype == torch.float16))

        # State
        self.global_step = 0
        self.start_epoch = 0
        self.best_loss = float("inf")

        # Total steps for scheduling
        self.steps_per_epoch = len(train_loader)
        self.total_steps = train_config.epochs * self.steps_per_epoch
        self.warmup_steps = train_config.warmup_epochs * self.steps_per_epoch

        logger.info(
            f"Trainer initialized: {self.total_steps} total steps, "
            f"{self.warmup_steps} warmup, AMP={self.use_amp} ({train_config.precision})"
        )

    def _build_optimizer(self) -> torch.optim.Optimizer:
        """Build AdamW optimizer with weight decay exclusion for norms/biases."""
        decay_params = []
        no_decay_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "norm" in name or "bias" in name or "cls_token" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        param_groups = [
            {"params": decay_params, "weight_decay": self.tc.weight_decay_start},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        # Add GradNorm parameters if present
        if self.criterion.gradnorm is not None:
            param_groups.append({
                "params": list(self.criterion.gradnorm.parameters()),
                "lr": self.mc.loss.gradnorm_lr,
                "weight_decay": 0.0,
            })

        return torch.optim.AdamW(
            param_groups,
            lr=self.tc.peak_lr,
            betas=(self.tc.beta1, self.tc.beta2),
            eps=self.tc.eps,
        )

    def _get_lr(self, step: int) -> float:
        """Cosine learning rate schedule with linear warmup."""
        if step < self.warmup_steps:
            return self.tc.peak_lr * step / max(self.warmup_steps, 1)
        progress = (step - self.warmup_steps) / max(self.total_steps - self.warmup_steps, 1)
        return self.tc.min_lr + 0.5 * (self.tc.peak_lr - self.tc.min_lr) * (1.0 + math.cos(math.pi * progress))

    def _get_weight_decay(self, step: int) -> float:
        """Cosine weight decay schedule: start -> end."""
        progress = step / max(self.total_steps, 1)
        return self.tc.weight_decay_start + 0.5 * (
            self.tc.weight_decay_end - self.tc.weight_decay_start
        ) * (1.0 - math.cos(math.pi * progress))

    def _update_lr_wd(self, step: int) -> tuple[float, float]:
        """Update optimizer learning rate and weight decay."""
        lr = self._get_lr(step)
        wd = self._get_weight_decay(step)

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
            if param_group.get("weight_decay", 0) > 0:
                param_group["weight_decay"] = wd

        return lr, wd

    def _mask_batch(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply masking strategy to batch."""
        loss_config = self.mc.loss
        mask_id = 1  # [MASK] token id
        vocab_size = self.mc.encoder.vocab_size

        if loss_config.mlm_mask_strategy == "span":
            return span_mask(
                tokens, mask_ratio=loss_config.mlm_mask_ratio,
                span_length=loss_config.mlm_span_length, mask_id=mask_id,
            )
        return random_mask(
            tokens, mask_ratio=loss_config.mlm_mask_ratio,
            mask_id=mask_id, vocab_size=vocab_size,
        )

    def _train_step(self, batch: dict) -> dict[str, float]:
        """Single training step: forward + backward + update."""
        tokens = batch["tokens"].to(self.device)
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        # Mask tokens
        masked_tokens, mask, labels = self._mask_batch(tokens)

        # Forward pass with AMP
        with torch.amp.autocast(device_type="cuda", dtype=self.amp_dtype, enabled=self.use_amp):
            model_output = self.model(tokens, masked_tokens, attention_mask)
            loss_output = self.criterion(model_output, labels)

        total_loss = loss_output["total"]

        # Backward
        self.optimizer.zero_grad(set_to_none=True)
        self.scaler.scale(total_loss).backward()

        # Gradient clipping
        if self.tc.grad_clip > 0:
            self.scaler.unscale_(self.optimizer)
            grad_norm = nn.utils.clip_grad_norm_(
                self.model.parameters(), self.tc.grad_clip
            )
        else:
            grad_norm = torch.tensor(0.0)

        self.scaler.step(self.optimizer)
        self.scaler.update()

        # Update target encoder
        if self.mc.loss.target_mode == "ema":
            decay = BJEPA.get_ema_decay(
                self.global_step, self.total_steps,
                self.mc.loss.ema_start, self.mc.loss.ema_end,
            )
            self.model.update_target_encoder(decay)
        else:
            self.model.update_target_encoder()

        # Collect metrics
        metrics = {
            "train/total_loss": total_loss.item(),
            "train/mlm_loss": loss_output["mlm"].item(),
            "train/jepa_loss": loss_output["jepa"].item(),
            "train/vicreg_var": loss_output["vicreg_var"].item(),
            "train/vicreg_cov": loss_output["vicreg_cov"].item(),
            "train/grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
        }

        # Add GradNorm weights if available
        for k, v in loss_output.items():
            if k.startswith("gradnorm"):
                metrics[f"train/{k}"] = v

        return metrics

    @torch.no_grad()
    def _evaluate(self, epoch: int) -> dict[str, float]:
        """Quick evaluation: RankMe + feature std on validation subset."""
        self.model.eval()
        loader = self.val_loader or self.train_loader

        all_embeddings = []
        total_loss = 0.0
        n_batches = 0
        max_eval_batches = 50

        for batch in loader:
            if n_batches >= max_eval_batches:
                break
            tokens = batch["tokens"].to(self.device)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)

            cls = self.model.encode(tokens, attention_mask, use_target=True)
            all_embeddings.append(cls.cpu())

            # Compute loss
            masked_tokens, mask_bool, labels = self._mask_batch(tokens)
            with torch.amp.autocast(device_type="cuda", dtype=self.amp_dtype, enabled=self.use_amp):
                out = self.model(tokens, masked_tokens, attention_mask)
                loss_out = self.criterion(out, labels)
            total_loss += loss_out["total"].item()
            n_batches += 1

        all_emb = torch.cat(all_embeddings, dim=0)

        metrics = {
            "eval/rankme": compute_rankme(all_emb),
            "eval/feature_std": compute_feature_std(all_emb),
            "eval/loss": total_loss / max(n_batches, 1),
            "eval/n_samples": len(all_emb),
        }

        self.model.train()
        return metrics

    def train(self) -> None:
        """Full training loop across all epochs."""
        logger.info(f"Starting training from epoch {self.start_epoch}")

        for epoch in range(self.start_epoch, self.tc.epochs):
            self.model.train()
            epoch_start = time.time()
            epoch_loss = 0.0
            n_steps = 0

            for batch in self.train_loader:
                # Update schedules
                lr, wd = self._update_lr_wd(self.global_step)

                # Training step
                metrics = self._train_step(batch)
                epoch_loss += metrics["train/total_loss"]
                n_steps += 1

                # Logging
                if self.global_step % self.tc.log_every == 0:
                    metrics["train/lr"] = lr
                    metrics["train/wd"] = wd
                    metrics["train/epoch"] = epoch
                    log_metrics(self.global_step, metrics, self.tc.use_wandb)

                    if self.global_step % (self.tc.log_every * 10) == 0:
                        logger.info(
                            f"Step {self.global_step} | epoch {epoch} | "
                            f"loss={metrics['train/total_loss']:.4f} | "
                            f"mlm={metrics['train/mlm_loss']:.4f} | "
                            f"jepa={metrics['train/jepa_loss']:.4f} | "
                            f"lr={lr:.2e}"
                        )

                self.global_step += 1

            epoch_time = time.time() - epoch_start
            avg_loss = epoch_loss / max(n_steps, 1)
            logger.info(f"Epoch {epoch} complete | avg_loss={avg_loss:.4f} | time={epoch_time:.1f}s")

            # Evaluation
            if (epoch + 1) % self.tc.eval_every == 0:
                eval_metrics = self._evaluate(epoch)
                eval_metrics["eval/epoch"] = epoch
                log_metrics(self.global_step, eval_metrics, self.tc.use_wandb)
                logger.info(
                    f"Eval | RankMe={eval_metrics['eval/rankme']:.1f} | "
                    f"std={eval_metrics['eval/feature_std']:.4f} | "
                    f"loss={eval_metrics['eval/loss']:.4f}"
                )

            # Checkpointing
            if (epoch + 1) % self.tc.save_every == 0:
                self._save_checkpoint(epoch, avg_loss)

        logger.info("Training complete!")

    def _save_checkpoint(self, epoch: int, loss: float) -> None:
        """Save checkpoint with full training state."""
        ckpt_dir = Path(self.tc.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        path = ckpt_dir / f"epoch{epoch:04d}.pt"
        torch.save({
            "epoch": epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "loss": loss,
        }, path)

        # Save best
        if loss < self.best_loss:
            self.best_loss = loss
            best_path = ckpt_dir / "best.pt"
            torch.save({
                "epoch": epoch,
                "global_step": self.global_step,
                "model_state_dict": self.model.state_dict(),
                "loss": loss,
            }, best_path)
            logger.info(f"New best model at epoch {epoch} (loss={loss:.4f})")

        log_checkpoint(str(path), {"epoch": epoch, "loss": loss})

    def resume(self, checkpoint_path: str) -> None:
        """Resume training from checkpoint."""
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scaler_state_dict" in ckpt:
            self.scaler.load_state_dict(ckpt["scaler_state_dict"])
        self.start_epoch = ckpt.get("epoch", 0) + 1
        self.global_step = ckpt.get("global_step", 0)
        self.best_loss = ckpt.get("loss", float("inf"))
        logger.info(
            f"Resumed from {checkpoint_path} | epoch={self.start_epoch} | step={self.global_step}"
        )
