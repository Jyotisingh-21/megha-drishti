import os
from pathlib import Path

import yaml


def load_thresholds(config_path: str | None = None) -> dict:
    if config_path is None:
        # Default to configs/thresholds.yaml in the project root
        root = Path(__file__).parent.parent.parent.parent
        config_path = root / "configs" / "thresholds.yaml"

    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_precip_thresholds() -> dict:
    return load_thresholds()["precip_mm_24h"]


def get_heatwave_thresholds() -> dict:
    return load_thresholds()["heatwave_celsius"]
