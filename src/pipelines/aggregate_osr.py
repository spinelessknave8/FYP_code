import os
import argparse
import csv
import json
import math


SPLITS = ["split_a", "split_b", "split_c"]
METRICS = [
    ("auroc_known_unknown", "AUROC"),
    ("tpr_unknown", "TPR_unknown"),
    ("fpr_known", "FPR_known"),
    ("open_set_acc", "OpenSetAcc"),
    ("known_accuracy", "KnownAcc"),
]


def mean_std(values):
    n = len(values)
    if n == 0:
        return float("nan"), float("nan")
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / n
    return mean, math.sqrt(var)


def main(output_dir: str):
    rows = []
    all_values = {k: [] for k, _ in METRICS}

    for split in SPLITS:
        path = os.path.join(output_dir, split, "osr", "metrics.json")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing metrics: {path}")
        with open(path, "r") as f:
            data = json.load(f)
        row = {"split": split}
        for key, label in METRICS:
            val = float(data[key])
            row[label] = val
            all_values[key].append(val)
        rows.append(row)

    # Mean ± std row
    mean_row = {"split": "mean"}
    std_row = {"split": "std"}
    for key, label in METRICS:
        m, s = mean_std(all_values[key])
        mean_row[label] = m
        std_row[label] = s
    rows.append(mean_row)
    rows.append(std_row)

    # CSV output
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "osr_summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["split"] + [label for _, label in METRICS])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    # Console table
    col_names = ["split"] + [label for _, label in METRICS]
    col_widths = {c: max(len(c), 10) for c in col_names}
    for r in rows:
        for c in col_names:
            val = r[c]
            if isinstance(val, float):
                s = f"{val:.4f}"
            else:
                s = str(val)
            col_widths[c] = max(col_widths[c], len(s))

    def fmt_row(r):
        parts = []
        for c in col_names:
            val = r[c]
            if isinstance(val, float):
                s = f"{val:.4f}"
            else:
                s = str(val)
            parts.append(s.ljust(col_widths[c]))
        return "  ".join(parts)

    print(fmt_row({c: c for c in col_names}))
    print("-" * (sum(col_widths.values()) + 2 * (len(col_names) - 1)))
    for r in rows:
        print(fmt_row(r))

    print(f"\nWrote: {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="outputs")
    args = parser.parse_args()
    main(args.output_dir)
