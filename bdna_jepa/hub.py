"""HuggingFace Hub integration: load/save/export B-JEPA models."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch

from bdna_jepa.config import BJEPAConfig, EncoderConfig, V31_CONFIG, V40_CONFIG
from bdna_jepa.models.encoder import TransformerEncoder
from bdna_jepa.models.jepa import BJEPA


VERSION_CONFIGS = {"v3.1": V31_CONFIG, "v4.0": V40_CONFIG}


def load_encoder(
    repo_or_path: str,
    version: str = "v4.0",
    device: str = "cpu",
) -> TransformerEncoder:
    """Load pretrained encoder from HuggingFace Hub or local checkpoint.

    Args:
        repo_or_path: HF repo ID (e.g. "orgava/dna-bacteria-jepa") or local path
        version: model version for config lookup
        device: target device

    Returns:
        TransformerEncoder with loaded weights
    """
    config = VERSION_CONFIGS.get(version, V40_CONFIG)

    path = Path(repo_or_path)
    if path.exists():
        checkpoint_path = path
    else:
        try:
            from huggingface_hub import hf_hub_download
            checkpoint_path = Path(hf_hub_download(
                repo_or_path, filename=f"encoder_{version}.pt"
            ))
        except ImportError:
            raise ImportError("Install huggingface-hub: pip install huggingface-hub")

    encoder = TransformerEncoder(config.encoder)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict):
        state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
        # Extract only context_encoder keys
        encoder_state = {}
        for k, v in state_dict.items():
            if k.startswith("context_encoder."):
                encoder_state[k.replace("context_encoder.", "")] = v
            elif k.startswith("target_encoder."):
                encoder_state[k.replace("target_encoder.", "")] = v
            elif not any(k.startswith(p) for p in ["predictor.", "mlm_head.", "fragment_pred."]):
                encoder_state[k] = v

        if encoder_state:
            encoder.load_state_dict(encoder_state, strict=False)
        else:
            encoder.load_state_dict(state_dict, strict=False)
    else:
        encoder.load_state_dict(ckpt, strict=False)

    return encoder.to(device).eval()


def load_full_model(
    checkpoint_path: str,
    config: Optional[BJEPAConfig] = None,
    version: str = "v4.0",
    device: str = "cpu",
) -> BJEPA:
    """Load full BJEPA model for continued training."""
    if config is None:
        config = VERSION_CONFIGS.get(version, V40_CONFIG)

    model = BJEPA(config)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"], strict=False)
    else:
        model.load_state_dict(ckpt, strict=False)

    return model.to(device)


def save_checkpoint(
    model: BJEPA,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict,
    path: str,
) -> None:
    """Save training checkpoint."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
    }, path)


def export_to_hub(
    model: BJEPA,
    repo_id: str,
    version: str = "v4.0",
    commit_message: str = "Upload B-JEPA model",
) -> None:
    """Export encoder weights to HuggingFace Hub."""
    try:
        from huggingface_hub import HfApi
    except ImportError:
        raise ImportError("Install huggingface-hub: pip install huggingface-hub")

    import tempfile
    api = HfApi()

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / f"encoder_{version}.pt"
        torch.save({
            "state_dict": model.context_encoder.state_dict(),
            "config": model.config.encoder.__dict__,
        }, path)

        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=f"encoder_{version}.pt",
            repo_id=repo_id,
            commit_message=commit_message,
        )
