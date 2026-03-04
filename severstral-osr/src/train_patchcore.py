import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipelines.two_stage.train_patchcore import main as train_patchcore_main  # noqa: E402


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    # train_patchcore uses repo configs/default.yaml merge logic.
    # For severstral-osr configs, pass absolute path and ensure it includes severstal settings.
    cfg_path = args.config
    if not os.path.isabs(cfg_path):
        cfg_path = str((ROOT / cfg_path).resolve())
    train_patchcore_main(cfg_path)
