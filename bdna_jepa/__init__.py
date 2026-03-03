"""B-JEPA: Bacterial Joint-Embedding Predictive Architecture."""

__version__ = "4.0.0"

from bdna_jepa.config import BJEPAConfig, EncoderConfig, V31_CONFIG, V40_CONFIG
from bdna_jepa.models.encoder import TransformerEncoder
from bdna_jepa.models.jepa import BJEPA
from bdna_jepa.hub import load_encoder

__all__ = [
    "BJEPA", "TransformerEncoder", "BJEPAConfig", "EncoderConfig",
    "V31_CONFIG", "V40_CONFIG", "load_encoder", "__version__",
]
