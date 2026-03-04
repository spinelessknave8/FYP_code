import argparse
import csv
from pathlib import Path

from common import load_config
from data import load_severstal_image_labels, collect_single_label_defect_samples, severstal_stats


def write_csv(path: Path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main(config_path: str):
    cfg = load_config(config_path)
    data_root = cfg["severstal"]["data_root"]
    train_csv = cfg["severstal"].get("train_csv", "train.csv")
    images_dir = cfg["severstal"].get("images_dir", "train_images")

    export_dir = Path(cfg.get("export_dir", "severstral-osr/exports"))
    labels = load_severstal_image_labels(data_root, train_csv)
    single = collect_single_label_defect_samples(data_root, train_csv, images_dir)
    stats = severstal_stats(data_root, train_csv, images_dir)

    # Full defect image labels (multi-label allowed)
    full_rows = []
    for image_id, cls_set in sorted(labels.items()):
        full_rows.append({
            "image_id": image_id,
            "classes": ";".join([f"Class_{c}" for c in sorted(cls_set)]),
            "num_classes": len(cls_set),
        })
    write_csv(export_dir / "all_defect_images.csv", ["image_id", "classes", "num_classes"], full_rows)

    # Single-label defect images
    single_rows = []
    for path, cls in single:
        single_rows.append({"image_id": Path(path).name, "class": cls, "path": path})
    write_csv(export_dir / "single_label_defect_images.csv", ["image_id", "class", "path"], single_rows)

    counts_rows = [
        {"class": f"Class_{k}", "single_label_image_count": v}
        for k, v in stats["single_label_images_per_class"].items()
    ]
    write_csv(export_dir / "single_label_defect_counts.csv", ["class", "single_label_image_count"], counts_rows)

    summary_rows = [{"key": k, "value": str(v)} for k, v in stats.items()]
    write_csv(export_dir / "severstal_summary.csv", ["key", "value"], summary_rows)

    print("Severstal classes:", [f"Class_{c}" for c in stats["defect_class_ids"]])
    print("Single-label counts:", {f"Class_{k}": v for k, v in stats["single_label_images_per_class"].items()})
    print("Wrote exports to:", export_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="severstral-osr/configs/default.yaml")
    args = parser.parse_args()
    main(args.config)
