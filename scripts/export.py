#!/usr/bin/env python3
"""Export B-JEPA checkpoint to HuggingFace Hub.

Usage:
    python scripts/export.py --checkpoint outputs/checkpoints/v4.0/best.pt --repo orgava/dna-bacteria-jepa
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bdna_jepa.hub import load_full_model, export_to_hub
from bdna_jepa.utils.logging import get_logger

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Export B-JEPA to HuggingFace Hub")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--repo", type=str, default="orgava/dna-bacteria-jepa")
    parser.add_argument("--version", type=str, default="v4.0")
    args = parser.parse_args()

    model = load_full_model(args.checkpoint, version=args.version)
    export_to_hub(model, args.repo, args.version)
    logger.info(f"Exported to {args.repo}")


if __name__ == "__main__":
    main()
