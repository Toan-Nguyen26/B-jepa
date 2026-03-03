#!/usr/bin/env python3
"""B-JEPA pretraining entry point.

Usage:
    python scripts/pretrain.py --config configs/training/v4.0.yaml
    python scripts/pretrain.py --config configs/training/v4.0.yaml --resume outputs/checkpoints/v4.0/epoch0100.pt
"""
from __future__ import annotations

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader

from bdna_jepa.config import BJEPAConfig, TrainingConfig, load_config, V40_CONFIG
from bdna_jepa.data.tokenizer import get_tokenizer
from bdna_jepa.data.dataset import BacterialGenomeDataset, collate_fn, GenomeAwareBatchSampler
from bdna_jepa.models.jepa import BJEPA
from bdna_jepa.losses.criterion import BJEPACriterion
from bdna_jepa.training.trainer import BJEPATrainer
from bdna_jepa.utils.logging import get_logger, setup_wandb


logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="B-JEPA Pretraining")
    parser.add_argument("--config", type=str, default=None, help="YAML config path")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint to resume from")
    parser.add_argument("--data", type=str, default=None, help="Override data path")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override peak LR")
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B logging")
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile")
    args = parser.parse_args()

    # Load config
    if args.config:
        model_config, train_config = load_config(args.config)
    else:
        model_config = V40_CONFIG
        train_config = TrainingConfig()

    # CLI overrides
    if args.data:
        train_config.data_path = args.data
    if args.epochs:
        train_config.epochs = args.epochs
    if args.batch_size:
        train_config.batch_size = args.batch_size
    if args.lr:
        train_config.peak_lr = args.lr
    if args.no_wandb:
        train_config.use_wandb = False
    if args.compile:
        train_config.compile_model = True

    # Seed
    torch.manual_seed(train_config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(train_config.seed)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Tokenizer
    version = "v4.0" if model_config.encoder.vocab_size > 128 else "v3.1"
    tokenizer = get_tokenizer(version, train_config.tokenizer_path)
    logger.info(f"Tokenizer: {type(tokenizer).__name__} (vocab={tokenizer.vocab_size})")

    # Dataset
    dataset = BacterialGenomeDataset(
        train_config.data_path, tokenizer, max_length=model_config.encoder.max_seq_len,
    )
    logger.info(f"Dataset: {len(dataset)} sequences")

    # DataLoader - use GenomeAwareBatchSampler if fragment JEPA enabled
    if model_config.loss.fragment.enabled and dataset.genome_to_indices:
        sampler = GenomeAwareBatchSampler(
            dataset.genome_to_indices,
            fragments_per_genome=model_config.loss.fragment.context_size,
            batch_size=train_config.batch_size,
        )
        train_loader = DataLoader(
            dataset, batch_sampler=sampler, collate_fn=collate_fn,
            num_workers=train_config.num_workers, pin_memory=True,
        )
    else:
        train_loader = DataLoader(
            dataset, batch_size=train_config.batch_size, shuffle=True,
            collate_fn=collate_fn, num_workers=train_config.num_workers,
            pin_memory=True, drop_last=True,
        )

    # Model
    model = BJEPA(model_config)
    n_params = sum(p.numel() for p in model.context_encoder.parameters())
    logger.info(f"Model: {n_params/1e6:.1f}M encoder params")

    if train_config.compile_model and hasattr(torch, "compile"):
        model = torch.compile(model)
        logger.info("Model compiled with torch.compile")

    # Criterion
    criterion = BJEPACriterion(model_config.loss)

    # W&B
    if train_config.use_wandb:
        from dataclasses import asdict
        config_dict = {
            "model": asdict(model_config),
            "training": asdict(train_config),
        }
        setup_wandb(train_config.wandb_project, config_dict, train_config.wandb_entity)

    # Trainer
    trainer = BJEPATrainer(
        model=model, criterion=criterion,
        train_loader=train_loader, val_loader=None,
        model_config=model_config, train_config=train_config,
        device=device,
    )

    # Resume
    if args.resume:
        trainer.resume(args.resume)

    # Train
    trainer.train()

    # Save final model
    model.save_weights(
        os.path.join(train_config.checkpoint_dir, "final.pt"),
        metadata={"epochs": train_config.epochs, "version": version},
    )


if __name__ == "__main__":
    main()
