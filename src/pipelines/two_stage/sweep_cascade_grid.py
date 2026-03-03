import os
import json
import argparse
import numpy as np
import csv


DEFAULT_SPLITS = ("split_a", "split_b", "split_c")


def compute_metrics(known_scores, unk_scores, conf_known, conf_unknown, tau, kappa, fusion_rule):
    known_no_defect = known_scores <= tau
    unk_no_defect = unk_scores <= tau

    if fusion_rule == "and":
        known_reject = (~known_no_defect) & (conf_known < kappa)
        unk_reject = (~unk_no_defect) & (conf_unknown < kappa)
    elif fusion_rule == "or":
        known_reject = (~known_no_defect) | (conf_known < kappa)
        unk_reject = (~unk_no_defect) | (conf_unknown < kappa)
    else:
        raise ValueError(f"Unsupported fusion_rule: {fusion_rule}")

    known_mask = ~known_no_defect
    unk_mask = ~unk_no_defect

    tpr_unknown_conditional = float(np.mean(unk_reject[unk_mask])) if np.sum(unk_mask) > 0 else float("nan")
    fpr_known_conditional = float(np.mean(known_reject[known_mask])) if np.sum(known_mask) > 0 else float("nan")
    tpr_unknown_system = float(np.mean(unk_reject))
    fpr_known_system = float(np.mean(known_reject))

    return {
        "stage1_pass_rate_known": float(np.mean(known_mask)),
        "stage1_pass_rate_unknown": float(np.mean(unk_mask)),
        "tpr_unknown_system": tpr_unknown_system,
        "fpr_known_system": fpr_known_system,
        "tpr_unknown_conditional": tpr_unknown_conditional,
        "fpr_known_conditional": fpr_known_conditional,
    }


def main(output_dir: str, kappas: str, rules: str, splits: str, out_csv: str):
    split_names = [s.strip() for s in splits.split(",") if s.strip()]
    kappa_values = [float(v.strip()) for v in kappas.split(",") if v.strip()]
    rule_values = [v.strip().lower() for v in rules.split(",") if v.strip()]

    rows = []
    for split in split_names:
        cdir = os.path.join(output_dir, split, "cascade")
        metrics_path = os.path.join(cdir, "metrics.json")
        if not os.path.exists(metrics_path):
            raise FileNotFoundError(f"Missing cascade metrics: {metrics_path}")

        with open(metrics_path, "r") as f:
            base_metrics = json.load(f)
        tau = float(base_metrics["patchcore_threshold"])
        base_kappa = float(base_metrics["kappa"])

        known_scores = np.load(os.path.join(cdir, "known_scores.npy"))
        unk_scores = np.load(os.path.join(cdir, "unknown_scores.npy"))
        conf_known = np.load(os.path.join(cdir, "known_conf.npy"))
        conf_unknown = np.load(os.path.join(cdir, "unknown_conf.npy"))

        for rule in rule_values:
            for kappa in kappa_values:
                m = compute_metrics(known_scores, unk_scores, conf_known, conf_unknown, tau, kappa, rule)
                rows.append({
                    "split": split,
                    "tau": tau,
                    "base_kappa": base_kappa,
                    "kappa": kappa,
                    "fusion_rule": rule,
                    **m,
                })

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote: {out_csv}")
    print("Rows:", len(rows))

    # Print compact mean summary by setting.
    by_key = {}
    for r in rows:
        key = (r["fusion_rule"], r["kappa"])
        by_key.setdefault(key, []).append(r)
    print("\nMean summary by (fusion_rule, kappa):")
    print("fusion_rule  kappa   tpr_unknown_system  fpr_known_system  pass_known  pass_unknown")
    for (rule, kappa), vals in sorted(by_key.items(), key=lambda x: (x[0][0], x[0][1])):
        mean_tpr = float(np.mean([v["tpr_unknown_system"] for v in vals]))
        mean_fpr = float(np.mean([v["fpr_known_system"] for v in vals]))
        mean_pk = float(np.mean([v["stage1_pass_rate_known"] for v in vals]))
        mean_pu = float(np.mean([v["stage1_pass_rate_unknown"] for v in vals]))
        print(f"{rule:<11} {kappa:<7.3f} {mean_tpr:<19.4f} {mean_fpr:<16.4f} {mean_pk:<10.4f} {mean_pu:<11.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--kappas", default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9")
    parser.add_argument("--rules", default="and,or")
    parser.add_argument("--splits", default="split_a,split_b,split_c")
    parser.add_argument("--out_csv", default="outputs/sweeps/cascade_grid/cascade_grid.csv")
    args = parser.parse_args()
    main(args.output_dir, args.kappas, args.rules, args.splits, args.out_csv)
