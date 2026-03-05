# Simulation Package

This folder gives you a **realistic synthetic benchmark** template for your FYP write-up.

## What it generates

- `simulated_metrics.csv`
  - one-stage vs two-stage quality and runtime metrics
- `simulated_roc_points.csv`
  - ROC points for defect-screening and open-set curves

## Scripts

1) Generate synthetic metrics/ROC CSVs:

```bash
python simulation/generate_simulation.py
```

2) Plot charts from generated CSVs (recommended in Colab runtime):

```bash
python simulation/plot_simulation.py
```

Plot outputs (when plotting script runs):
- `simulation/plots/roc_defect_screening.png`
- `simulation/plots/roc_open_set.png`
- `simulation/plots/quality_bars.png`
- `simulation/plots/runtime_bars.png`

## Note

These are simulated numbers for planning/presentation only. They are **not** experimental results.
