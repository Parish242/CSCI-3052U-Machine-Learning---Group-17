"""Robustness check for the paired bootstrap: resample whole Background passages instead of items.

The 521 test items come from 63 Background passages (about 8 items each), so items are not
independent. This script recomputes the 95% CI of every difference in bootstrap_differences.csv
with a cluster bootstrap over Backgrounds and writes both intervals side by side.

Run from the repository root: python src/cluster_bootstrap_check.py
Writes: results/cluster_bootstrap_check.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from data_pipeline import get_grouped_splits
from paired_bootstrap import LABELS, load_and_align

RESULTS = Path(__file__).resolve().parents[1] / "results"
N_BOOT = 10000
SEED = 42


def encode(values):
    # yes/no/neither -> 0/1/2; anything else (unparseable) -> 3, which counts as an error
    mapping = {label: i for i, label in enumerate(LABELS)}
    return np.array([mapping.get(v, 3) for v in values])


def macro_f1_from_codes(gold, pred):
    """Macro-F1 over the three labels, same as sklearn f1_score(labels=LABELS, zero_division=0)."""
    cm = np.bincount(gold * 4 + pred, minlength=16).reshape(4, 4)
    scores = []
    for c in range(3):
        tp = cm[c, c]
        denom = 2 * tp + (cm[:, c].sum() - tp) + (cm[c, :].sum() - tp)
        scores.append(0.0 if denom == 0 else 2 * tp / denom)
    return float(np.mean(scores))


def main():
    data = load_and_align()
    _, test, _ = get_grouped_splits()
    background = dict(zip(test["ID"].astype(str), test["Background"]))
    data["Background"] = data["ID"].map(background)
    assert data["Background"].notna().all()

    gold = encode(data["gold"])
    diffs = pd.read_csv(RESULTS / "bootstrap_differences.csv")
    models = sorted(set(diffs["model_a"]) | set(diffs["model_b"]))
    codes = {m: encode(data[m]) for m in models}

    # The fast scorer must agree with sklearn before it is trusted
    for m in models:
        expected = f1_score(data["gold"], data[m], labels=LABELS, average="macro", zero_division=0)
        assert abs(macro_f1_from_codes(gold, codes[m]) - expected) < 1e-12, m

    groups = list(data.groupby("Background").indices.values())
    rng = np.random.default_rng(SEED)
    boot = {m: np.empty(N_BOOT) for m in models}
    for i in range(N_BOOT):
        idx = np.concatenate([groups[k] for k in rng.integers(0, len(groups), len(groups))])
        for m in models:
            boot[m][i] = macro_f1_from_codes(gold[idx], codes[m][idx])

    rows = []
    for _, r in diffs.iterrows():
        d = boot[r["model_a"]] - boot[r["model_b"]]
        lo, hi = np.percentile(d, [2.5, 97.5])
        rows.append({
            "model_a": r["model_a"], "model_b": r["model_b"],
            "difference_a_minus_b": r["difference_a_minus_b"],
            "item_ci_lower": r["ci_lower"], "item_ci_upper": r["ci_upper"],
            "cluster_ci_lower": lo, "cluster_ci_upper": hi,
            "max_ci_bound_shift": max(abs(lo - r["ci_lower"]), abs(hi - r["ci_upper"])),
            "clusters": len(groups), "resamples": N_BOOT,
        })
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "cluster_bootstrap_check.csv", index=False)
    print(out.round(3).to_string(index=False))
    print(f"\nLargest CI bound shift: {out['max_ci_bound_shift'].max():.4f}")
    print("Saved results/cluster_bootstrap_check.csv")


if __name__ == "__main__":
    main()
