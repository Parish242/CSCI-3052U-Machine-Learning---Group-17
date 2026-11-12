
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

LABELS = ["yes", "no", "neither"]
N_BOOT = 10000
SEED = 42

LLM_FILES = {
    "InternLM | Without RoT":
        "internlm2.5-7b/full/internlm2.5-7b_without-rule.csv",
    "InternLM | With RoT":
        "internlm2.5-7b/full/internlm2.5-7b_with-rule.csv",
    "Gemma | Without RoT":
        "gemma-2-27b/full/gemma-2-27b_full_without_rule.csv",
    "Gemma | With RoT":
        "gemma-2-27b/full/gemma-2-27b_full_with_rule.csv",
}


def normalize(series):
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .replace({"neutral": "neither"})
    )


def load_and_align():
    classical = pd.read_csv(
        RESULTS / "classical_test_predictions.csv"
    )

    classical["ID"] = classical["ID"].astype(str)
    classical["gold"] = normalize(classical["gold"])
    classical["pred"] = normalize(classical["pred"])

    merged = None
    for (model, condition), group in classical.groupby(
        ["model", "condition"]
    ):
        suffix = (
            "With RoT"
            if condition == "With Rule-of-Thumb"
            else "Without RoT"
        )
        label = f"{model} | {suffix}"

        part = group[["ID", "gold", "pred"]].copy()
        assert part["ID"].is_unique
        part = part.rename(
            columns={"pred": label}
        )

        if merged is None:
            merged = part
        else:
            merged = merged.merge(
                part[["ID", label]],
                on="ID",
                validate="one_to_one",
            )

    # Align all LLM predictions by test ID
    for label, path in LLM_FILES.items():
        llm = pd.read_csv(RESULTS / path)
        llm["ID"] = llm["ID"].astype(str)
        llm["gold"] = normalize(llm["gold"])
        llm["pred"] = normalize(llm["pred"])

        assert llm["ID"].is_unique

        part = llm[["ID", "gold", "pred"]].rename(
            columns={
                "gold": "llm_gold",
                "pred": label,
            }
        )

        merged = merged.merge(
            part, on="ID", validate="one_to_one"
        )

        assert (
            merged["gold"] == merged["llm_gold"]
        ).all(), f"Gold labels disagree for {label}"

        merged = merged.drop(columns="llm_gold")

    assert len(merged) == 521, (
        f"Expected 521 shared IDs; got {len(merged)}"
    )
    assert merged["ID"].is_unique
    assert merged["gold"].isin(LABELS).all()

    return merged.sort_values("ID").reset_index(drop=True)


def macro_f1(y_true, y_pred):
    return f1_score(
        y_true, y_pred,
        labels=LABELS,
        average="macro",
        zero_division=0,
    )


def main():
    data = load_and_align()
    gold = data["gold"].to_numpy()
    models = [
        col for col in data.columns
        if col not in ("ID", "gold")
    ]
    predictions = {
        name: data[name].to_numpy()
        for name in models
    }

    # Check point estimates against the recorded scores
    print("Original test macro-F1:")
    observed = {}
    for name in models:
        observed[name] = macro_f1(
            gold, predictions[name]
        )
        print(f"{name}: {observed[name]:.4f}")

    rng = np.random.default_rng(SEED)
    boot = {
        name: np.empty(N_BOOT)
        for name in models
    }

    # Exactly the same indices for every model
    for i in range(N_BOOT):
        indices = rng.integers(
            0, len(data), size=len(data)
        )
        sample_gold = gold[indices]

        for name in models:
            boot[name][i] = macro_f1(
                sample_gold,
                predictions[name][indices],
            )

        if (i + 1) % 1000 == 0:
            print(f"Completed {i + 1}/{N_BOOT}")

    ci_rows = []
    for name in models:
        lower, upper = np.percentile(
            boot[name], [2.5, 97.5]
        )
        ci_rows.append({
            "model": name,
            "macro_f1": observed[name],
            "ci_lower": lower,
            "ci_upper": upper,
            "resamples": N_BOOT,
        })

    pairs = [
        (
            "Logistic Regression | Without RoT",
            "Naive Bayes | Without RoT",
        ),
        (
            "Logistic Regression | With RoT",
            "Naive Bayes | With RoT",
        ),
        (
            "Logistic Regression | With RoT",
            "Logistic Regression | Without RoT",
        ),
        (
            "Naive Bayes | With RoT",
            "Naive Bayes | Without RoT",
        ),
        (
            "InternLM | With RoT",
            "InternLM | Without RoT",
        ),
        (
            "Gemma | With RoT",
            "Gemma | Without RoT",
        ),
        (
            "Logistic Regression | Without RoT",
            "InternLM | With RoT",
        ),
    ]

    diff_rows = []
    for first, second in pairs:
        differences = boot[first] - boot[second]
        lower, upper = np.percentile(
            differences, [2.5, 97.5]
        )
        diff_rows.append({
            "model_a": first,
            "model_b": second,
            "difference_a_minus_b":
                observed[first] - observed[second],
            "ci_lower": lower,
            "ci_upper": upper,
            "resamples": N_BOOT,
        })

    pd.DataFrame(ci_rows).to_csv(
        RESULTS / "bootstrap_ci.csv",
        index=False,
    )
    pd.DataFrame(diff_rows).to_csv(
        RESULTS / "bootstrap_differences.csv",
        index=False,
    )

    print("\nModel confidence intervals:")
    print(pd.DataFrame(ci_rows).to_string(index=False))

    print("\nPaired differences:")
    print(pd.DataFrame(diff_rows).to_string(index=False))


if __name__ == "__main__":
    main()
