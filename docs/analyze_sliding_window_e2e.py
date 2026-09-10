"""Regenerate the sliding-window training table from the 24 recorded run medians.

Requires NumPy. Percentages are candidate-minus-baseline step time, divided by
the observed SWA baseline median. Standard errors are conditional on each OLS
model, including its shared linear position drift and independent residuals.
"""

import argparse
import json
from pathlib import Path

import numpy as np


def fit(rows, mode, denominator):
    arms = ["both", "bwdonly", "ctlF_base", "ctlF_both", "fwdonly"]
    jobs = sorted({row["job"] for row in rows})
    if mode == "drop_first":
        rows = [row for row in rows if row["position"] != 1]
    x, y = [], []
    for row in rows:
        design = [1.] + [float(row["arm"] == arm) for arm in arms]
        design += [float(row["position"]), float(row["job"] == jobs[-1])]
        if mode == "cold_start":
            design += [float(row["position"] == 1)]
        x.append(design)
        y.append(row["dt_ms"])
    x, y = np.array(x), np.array(y)
    beta, _, rank, _ = np.linalg.lstsq(x, y, rcond=None)
    if rank != x.shape[1] or len(y) <= rank:
        raise ValueError("Run order does not identify the model with residual degrees of freedom")
    dof = len(y) - rank
    residual = y - x @ beta
    covariance = (residual @ residual / dof) * np.linalg.inv(x.T @ x)
    estimates = {}
    for arm in ("fwdonly", "bwdonly", "both", "control"):
        contrast = np.zeros(len(beta))
        if arm == "control":
            contrast[1 + arms.index("ctlF_both")] = 1
            contrast[1 + arms.index("ctlF_base")] = -1
        else:
            contrast[1 + arms.index(arm)] = 1
        delta = float(contrast @ beta)
        # The control is a within-model contrast; retain the covariance term.
        se = float(np.sqrt(contrast @ covariance @ contrast))
        estimates[arm] = dict(
            percent=100 * delta / denominator,
            se_percent=100 * se / denominator,
            t=delta / se,
        )
    return estimates, len(y), int(dof)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs", type=Path,
        default=Path(__file__).parent / "data/sliding_window_e2e_runs.json",
    )
    args = parser.parse_args()
    rows = json.loads(args.runs.read_text())["runs"]
    denominator = float(np.median([row["dt_ms"] for row in rows if row["arm"] == "base"]))
    print(f"Observed SWA baseline median: {denominator:.4f} ms")
    print("Time change: negative = faster; positive = slower. Control = ctlF_both - ctlF_base.")
    print("Standard errors are conditional on the specified OLS model.\n")
    print("| Model | Forward | Backward | Both | All-full control |")
    print("|---|---:|---:|---:|---:|")
    fitted = []
    for mode, label in [
        ("linear", "Without cold-start term"),
        ("cold_start", "With cold-start term"),
        ("drop_first", "Drop first position"),
    ]:
        estimates, n, dof = fit(rows, mode, denominator)
        cells = " | ".join(f"{value['percent']:+.2f}%" for value in estimates.values())
        print(f"| {label} | {cells} |")
        fitted.append((label, estimates, n, dof))
    print()
    for label, estimates, n, dof in fitted:
        print(f"{label}: n={n}, residual dof={dof}")
        for arm, value in estimates.items():
            print(
                f"  {arm}: {value['percent']:+.4f}%, "
                f"SE={value['se_percent']:.4f} percentage points, t={value['t']:+.4f}"
            )


if __name__ == "__main__":
    main()
