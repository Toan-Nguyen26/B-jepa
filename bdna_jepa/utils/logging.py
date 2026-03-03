"""Logging utilities: W&B integration + console formatting."""
from __future__ import annotations

import logging
import sys
from typing import Optional


def get_logger(name: str = "bdna_jepa") -> logging.Logger:
    """Get a configured logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def setup_wandb(
    project: str,
    config: dict,
    entity: Optional[str] = None,
    run_name: Optional[str] = None,
    resume_id: Optional[str] = None,
) -> Optional[object]:
    """Initialize W&B run. Returns run object or None if wandb unavailable."""
    try:
        import wandb
        kwargs = {"project": project, "config": config}
        if entity:
            kwargs["entity"] = entity
        if run_name:
            kwargs["name"] = run_name
        if resume_id:
            kwargs["id"] = resume_id
            kwargs["resume"] = "allow"
        return wandb.init(**kwargs)
    except ImportError:
        get_logger().warning("wandb not installed, logging to console only")
        return None


def log_metrics(step: int, metrics: dict, use_wandb: bool = True) -> None:
    """Log metrics to W&B and console."""
    try:
        if use_wandb:
            import wandb
            if wandb.run is not None:
                wandb.log(metrics, step=step)
    except ImportError:
        pass


def log_checkpoint(path: str, metrics: dict) -> None:
    """Log checkpoint save event."""
    logger = get_logger()
    summary = ", ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in metrics.items())
    logger.info(f"Saved checkpoint: {path} | {summary}")
