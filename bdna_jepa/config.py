"""Model configuration dataclasses for all B-JEPA versions."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Literal, Optional

import yaml


@dataclass
class EncoderConfig:
    vocab_size: int = 4096
    embed_dim: int = 576
    num_layers: int = 12
    num_heads: int = 9
    ff_dim: int = 2304
    ff_activation: Literal["gelu", "swiglu"] = "swiglu"
    dropout: float = 0.1
    max_seq_len: int = 1024
    pos_encoding: Literal["learned", "rotary"] = "rotary"
    norm_type: Literal["layernorm", "rmsnorm"] = "rmsnorm"
    qk_norm: bool = True
    attention_dropout: float = 0.0
    embed_dropout: float = 0.1
    bias: bool = False

    @property
    def head_dim(self) -> int:
        return self.embed_dim // self.num_heads


@dataclass
class PredictorConfig:
    depth: int = 4
    dim: int = 192
    num_heads: int = 6
    ff_dim: int = 768
    ff_activation: Literal["gelu", "swiglu"] = "swiglu"
    dropout: float = 0.1
    norm_type: Literal["layernorm", "rmsnorm"] = "rmsnorm"
    bias: bool = False


@dataclass
class FragmentConfig:
    enabled: bool = True
    context_size: int = 4
    predictor_depth: int = 2
    predictor_dim: int = 192
    predictor_heads: int = 6
    loss_weight: float = 1.0


@dataclass
class LossConfig:
    target_mode: Literal["stop_grad", "ema"] = "stop_grad"
    ema_start: float = 0.996
    ema_end: float = 1.0
    ema_schedule: Literal["cosine", "linear"] = "cosine"

    weight_mlm: float = 1.0
    weight_jepa: float = 1.0
    weight_vicreg_var: float = 25.0
    weight_vicreg_cov: float = 1.0
    weight_fragment: float = 1.0

    vicreg_gamma: float = 1.0
    jepa_loss_type: Literal["smooth_l1", "mse", "cosine"] = "smooth_l1"

    use_gradnorm: bool = True
    gradnorm_alpha: float = 1.5
    gradnorm_lr: float = 0.025

    mlm_mask_ratio: float = 0.15
    mlm_mask_strategy: Literal["random", "span"] = "span"
    mlm_span_length: int = 5

    fragment: FragmentConfig = field(default_factory=FragmentConfig)


@dataclass
class TrainingConfig:
    optimizer: Literal["adamw"] = "adamw"
    peak_lr: float = 1e-3
    min_lr: float = 1e-6
    warmup_epochs: int = 20
    weight_decay_start: float = 0.04
    weight_decay_end: float = 0.4
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    grad_clip: float = 1.0
    lr_schedule: Literal["cosine", "linear"] = "cosine"

    epochs: int = 300
    batch_size: int = 256
    num_workers: int = 4
    mixed_precision: bool = True
    precision: Literal["bf16", "fp16"] = "bf16"
    compile_model: bool = False

    data_path: str = "data/processed/pretrain_sequences_expanded.csv"
    tokenizer_path: Optional[str] = None

    checkpoint_dir: str = "outputs/checkpoints/v4.0"
    save_every: int = 10
    eval_every: int = 10
    log_every: int = 50

    use_wandb: bool = True
    wandb_project: str = "bdna-jepa"
    wandb_entity: Optional[str] = None

    seed: int = 42


@dataclass
class BJEPAConfig:
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    predictor: PredictorConfig = field(default_factory=PredictorConfig)
    loss: LossConfig = field(default_factory=LossConfig)


V31_CONFIG = BJEPAConfig(
    encoder=EncoderConfig(
        vocab_size=128, embed_dim=384, num_layers=6, num_heads=6,
        ff_dim=1024, ff_activation="gelu", max_seq_len=512,
        pos_encoding="learned", norm_type="layernorm", qk_norm=False, bias=True,
    ),
    predictor=PredictorConfig(
        depth=4, dim=384, num_heads=6, ff_dim=1024,
        ff_activation="gelu", norm_type="layernorm", bias=True,
    ),
    loss=LossConfig(
        target_mode="stop_grad", use_gradnorm=False, weight_vicreg_var=10.0,
        fragment=FragmentConfig(enabled=False),
    ),
)

V40_CONFIG = BJEPAConfig(
    encoder=EncoderConfig(
        vocab_size=4096, embed_dim=576, num_layers=12, num_heads=9,
        ff_dim=2304, ff_activation="swiglu", max_seq_len=1024,
        pos_encoding="rotary", norm_type="rmsnorm", qk_norm=True, bias=False,
    ),
    predictor=PredictorConfig(
        depth=4, dim=192, num_heads=6, ff_dim=768,
        ff_activation="swiglu", norm_type="rmsnorm", bias=False,
    ),
    loss=LossConfig(
        target_mode="stop_grad", use_gradnorm=True, gradnorm_alpha=1.5,
        weight_vicreg_var=25.0, weight_vicreg_cov=1.0,
        jepa_loss_type="smooth_l1", mlm_mask_strategy="span", mlm_span_length=5,
        fragment=FragmentConfig(enabled=True, context_size=4),
    ),
)


def load_config(path: str | Path) -> tuple[BJEPAConfig, TrainingConfig]:
    path = Path(path)
    with open(path) as f:
        raw = yaml.safe_load(f)

    enc_raw = raw.get("encoder", {})
    encoder = EncoderConfig(**{k: v for k, v in enc_raw.items() if k in EncoderConfig.__dataclass_fields__})

    pred_raw = raw.get("predictor", {})
    predictor = PredictorConfig(**{k: v for k, v in pred_raw.items() if k in PredictorConfig.__dataclass_fields__})

    loss_raw = raw.get("loss", {})
    frag_raw = loss_raw.pop("fragment", {}) if "fragment" in loss_raw else {}
    fragment = FragmentConfig(**{k: v for k, v in frag_raw.items() if k in FragmentConfig.__dataclass_fields__})
    loss = LossConfig(
        fragment=fragment,
        **{k: v for k, v in loss_raw.items() if k in LossConfig.__dataclass_fields__},
    )

    model_config = BJEPAConfig(encoder=encoder, predictor=predictor, loss=loss)
    train_raw = raw.get("training", raw)
    training = TrainingConfig(**{k: v for k, v in train_raw.items() if k in TrainingConfig.__dataclass_fields__})
    return model_config, training


def save_config(model_config: BJEPAConfig, training_config: TrainingConfig, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "encoder": asdict(model_config.encoder),
        "predictor": asdict(model_config.predictor),
        "loss": asdict(model_config.loss),
        "training": asdict(training_config),
    }
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
