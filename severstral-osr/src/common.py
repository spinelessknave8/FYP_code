import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.io import load_yaml, deep_update  # noqa: E402


def load_config(config_path: str) -> dict:
    default_path = ROOT / "severstral-osr" / "configs" / "default.yaml"
    cfg = load_yaml(str(default_path))
    if os.path.basename(config_path) != "default.yaml":
        cfg = deep_update(cfg, load_yaml(config_path))
    return cfg


def split_name_from_config(config_path: str) -> str:
    name = os.path.splitext(os.path.basename(config_path))[0]
    if name.endswith(".colab"):
        name = name[: -len(".colab")]
    return name
