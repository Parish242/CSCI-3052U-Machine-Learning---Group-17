"""Build the main results table and figures.

Reads: whichever result files exist (no retraining or rerunning)

Writes:
  results/main_table.csv, results/main_table.md
  results/figures/fig1_test_macro_f1.png      test macro-F1 with 95% bootstrap CIs
  results/figures/fig2_question_ablation.png  drop in macro-F1 when the question is removed
"""
from pathlib import Path

import pandas as pd

import plot_style
from plot_style import plt

RESULTS = Path(__file__).resolve().parents[1] / "results"
FIGURES = RESULTS / "figures"

MODELS = [
    # key, display name, family
    ("Logistic Regression", "TF-IDF + Logistic Regression", "classical"),
    ("Naive Bayes", "TF-IDF + Naive Bayes", "classical"),
    ("InternLM", "InternLM2.5-7B-chat (8-bit)", "llm"),
    ("Gemma", "Gemma-2-27B-it (4-bit, v1 prompt)", "llm"),
]
LLM_KEYS = {"internlm2.5-7b": "InternLM", "gemma-2-27b": "Gemma"}
CONDITIONS = ["Without Rule-of-Thumb", "With Rule-of-Thumb"]


def read(name):
    path = RESULTS / name
    if not path.exists():
        print(f"Skipping results/{name}: not found")
        return None
    return pd.read_csv(path)


def boot_label(key, condition, question):
    label = f"{key} | {'With RoT' if condition == 'With Rule-of-Thumb' else 'Without RoT'}"
    return label + (" | No Question" if question == "removed" else "")


def collect_rows():
    rows = []

    # Classical models from the baseline tables
    for name, question in [("baseline_results.csv", "kept"), ("baseline_results_no_question.csv", "removed")]:
        df = read(name)
        if df is None:
            continue
        for _, r in df.iterrows():
            rows.append({
                "key": r["model"],
                "condition": r["condition"],
                "question": question,
                "cv_macro_f1_mean": r.get("cv_macro_f1"),
                "cv_macro_f1_std": r.get("cv_macro_f1_std"),
                "test_accuracy": r["accuracy"],
                "test_macro_f1": r["macro_f1"],
                "seed_std": r.get("macro_f1_seed_std"),
                "source": f"results/{name}",
            })

    # LLMs: test scores plus mean/std over the same 5 CV folds
    test = read("llm_test_scores.csv")
    cv = read("llm_cv_scores.csv")
    if test is not None:
        cv_agg = None
        if cv is not None:
            cv_agg = cv.groupby(["model", "condition", "question"])["macro_f1"].agg(["mean", "std"])
        for _, r in test.iterrows():
            cv_row = cv_agg.loc[(r["model"], r["condition"], r["question"])] if cv_agg is not None else None
            rows.append({
                "key": LLM_KEYS.get(r["model"], r["model"]),
                "condition": r["condition"],
                "question": r["question"],
                "cv_macro_f1_mean": None if cv_row is None else cv_row["mean"],
                "cv_macro_f1_std": None if cv_row is None else cv_row["std"],
                "test_accuracy": r["accuracy"],
                "test_macro_f1": r["macro_f1"],
                "unparseable": r["unparseable"],
                "source": "results/llm_test_scores.csv",
            })

    table = pd.DataFrame(rows)

    # 95% bootstrap CIs on the test set
    ci = read("bootstrap_ci.csv")
    table["test_ci_lower"] = None
    table["test_ci_upper"] = None
    if ci is not None:
        ci = ci.set_index("model")
        for i, r in table.iterrows():
            label = boot_label(r["key"], r["condition"], r["question"])
            if label in ci.index:
                table.at[i, "test_ci_lower"] = ci.at[label, "ci_lower"]
                table.at[i, "test_ci_upper"] = ci.at[label, "ci_upper"]

    names = {k: d for k, d, _ in MODELS}
    names.update({
        "Dummy Classifier": "Dummy (most frequent)",
        "Dummy Classifier (stratified)": "Dummy (stratified, 10 seeds)",
    })
    order = ["Dummy Classifier", "Dummy Classifier (stratified)"] + [k for k, _, _ in MODELS]
    table["model"] = table["key"].map(names).fillna(table["key"])
    table["_m"] = table["key"].map({k: i for i, k in enumerate(order)}).fillna(len(order))
    table["_c"] = table["condition"].map({c: i for i, c in enumerate(CONDITIONS)})
    table["_q"] = table["question"].map({"kept": 0, "removed": 1})
    is_dummy = table["key"].str.startswith("Dummy")
    table.loc[is_dummy, "condition"] = "N/A"
    table = table.sort_values(["_q", "_m", "_c"]).drop(columns=["_m", "_c", "_q"]).reset_index(drop=True)
    return table


def fmt(x, nd=4):
    return "" if pd.isna(x) else f"{x:.{nd}f}"


def write_tables(table):
    cols = ["model", "condition", "question", "cv_macro_f1_mean", "cv_macro_f1_std",
            "test_accuracy", "test_macro_f1", "test_ci_lower", "test_ci_upper",
            "seed_std", "unparseable", "source"]
    table[[c for c in cols if c in table.columns]].to_csv(RESULTS / "main_table.csv", index=False)

    lines = [
        "| Model | Rule-of-Thumb | Question | CV Macro-F1 (mean ± SD, 5 folds) | Test Accuracy | Test Macro-F1 [95% CI] |",
        "|---|---|---|---:|---:|---:|",
    ]
    for _, r in table.iterrows():
        cv = "—" if pd.isna(r["cv_macro_f1_mean"]) else f"{fmt(r['cv_macro_f1_mean'])} ± {fmt(r['cv_macro_f1_std'])}"
        f1 = fmt(r["test_macro_f1"])
        if not pd.isna(r.get("test_ci_lower")):
            f1 += f" [{fmt(r['test_ci_lower'])}, {fmt(r['test_ci_upper'])}]"
        elif not pd.isna(r.get("seed_std")):
            f1 += f" ± {fmt(r['seed_std'])} (seed SD)"
        rule = r["condition"].replace(" Rule-of-Thumb", "")
        lines.append(f"| {r['model']} | {rule} | {r['question']} | {cv} | {fmt(r['test_accuracy'])} | {f1} |")
    lines += [
        "",
        "Test set: 521 held-out items (grouped by Background). CV: the same 5 StratifiedGroupKFold folds for every model.",
        "95% CIs: percentile bootstrap over test items, 10,000 resamples (src/paired_bootstrap.py).",
        "Gemma uses a different prompt from InternLM (v1, see results/gemma-2-27b/full/gemma-2-27b_run-notes_full.txt) "
        "and was not run with the question removed.",
    ]
    (RESULTS / "main_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def dot_ci(ax, y, row, color):
    x = row["test_macro_f1"]
    if not pd.isna(row["test_ci_lower"]):
        ax.hlines(y, row["test_ci_lower"], row["test_ci_upper"], color=color, linewidth=2, zorder=2)
    ax.plot(x, y, "o", color=color, markersize=8, markeredgecolor=plot_style.SURFACE,
            markeredgewidth=1.5, zorder=3)


def fig_main(table):
    kept = table[table["question"] == "kept"]
    models = [(k, d) for k, d, _ in MODELS if (kept["key"] == k).any()]
    if not models:
        return
    fig, ax = plt.subplots(figsize=(7.2, 0.75 * len(models) + 1.6))
    offset = 0.17
    for i, (key, display) in enumerate(models):
        y0 = len(models) - 1 - i
        for cond, color, dy in [(CONDITIONS[0], plot_style.WITHOUT_ROT, offset),
                                (CONDITIONS[1], plot_style.WITH_ROT, -offset)]:
            r = kept[(kept["key"] == key) & (kept["condition"] == cond)]
            if len(r):
                dot_ci(ax, y0 + dy, r.iloc[0], color)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([d for _, d in reversed(models)])
    ax.tick_params(axis="y", length=0)

    # Reference floors from the dummy baselines
    for key, label in [("Dummy Classifier (stratified)", "stratified guess"),
                       ("Dummy Classifier", "always 'yes'")]:
        r = kept[kept["key"] == key]
        if len(r):
            x = r.iloc[0]["test_macro_f1"]
            ax.axvline(x, color=plot_style.TEXT_MUTED, linewidth=1, linestyle=(0, (3, 3)), zorder=1)
            ax.text(x + 0.008, len(models) - 0.45, f"{label}\n{x:.2f}", fontsize=8,
                    color=plot_style.TEXT_SECONDARY, va="top")

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.6, len(models) - 0.4)
    ax.set_xlabel("Test macro-F1 (521 items), dots = point estimate, lines = 95% bootstrap CI")
    ax.grid(axis="x", color=plot_style.GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    handles = [
        plt.Line2D([], [], marker="o", color=plot_style.WITHOUT_ROT, markersize=8, linewidth=2, label="Without rule-of-thumb"),
        plt.Line2D([], [], marker="o", color=plot_style.WITH_ROT, markersize=8, linewidth=2, label="With rule-of-thumb"),
    ]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, handlelength=2.2)
    ax.set_title("Figure 1. Test macro-F1 by model and condition (question kept)", pad=28)
    fig.savefig(FIGURES / "fig1_test_macro_f1.png")
    plt.close(fig)
    print("Saved results/figures/fig1_test_macro_f1.png")


def fig_ablation():
    diffs = read("bootstrap_differences.csv")
    if diffs is None:
        return
    rows = []
    for key, display, _ in MODELS:
        for cond in CONDITIONS:
            kept, removed = boot_label(key, cond, "kept"), boot_label(key, cond, "removed")
            hit = diffs[(diffs["model_a"] == kept) & (diffs["model_b"] == removed)]
            if len(hit):
                rows.append((key, display, cond, hit.iloc[0]))
    if not rows:
        return
    models = list(dict.fromkeys((k, d) for k, d, _, _ in rows))
    fig, ax = plt.subplots(figsize=(7.2, 0.75 * len(models) + 1.6))
    offset = 0.17
    for key, display, cond, r in rows:
        y0 = len(models) - 1 - models.index((key, display))
        color, dy = ((plot_style.WITHOUT_ROT, offset) if cond == CONDITIONS[0]
                     else (plot_style.WITH_ROT, -offset))
        ax.hlines(y0 + dy, r["ci_lower"], r["ci_upper"], color=color, linewidth=2, zorder=2)
        ax.plot(r["difference_a_minus_b"], y0 + dy, "o", color=color, markersize=8,
                markeredgecolor=plot_style.SURFACE, markeredgewidth=1.5, zorder=3)
    ax.axvline(0, color=plot_style.TEXT_SECONDARY, linewidth=1, zorder=1)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([d for _, d in reversed(models)])
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(-0.6, len(models) - 0.4)
    ax.set_xlabel("Drop in test macro-F1 when the final question is removed\n"
                  "(kept − removed, paired bootstrap 95% CI; right of 0 = the question helps)")
    ax.grid(axis="x", color=plot_style.GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    handles = [
        plt.Line2D([], [], marker="o", color=plot_style.WITHOUT_ROT, markersize=8, linewidth=2, label="Without rule-of-thumb"),
        plt.Line2D([], [], marker="o", color=plot_style.WITH_ROT, markersize=8, linewidth=2, label="With rule-of-thumb"),
    ]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, handlelength=2.2)
    ax.set_title("Figure 2. Question ablation (Gemma not run with the question removed)", pad=28)
    fig.savefig(FIGURES / "fig2_question_ablation.png")
    plt.close(fig)
    print("Saved results/figures/fig2_question_ablation.png")


def main():
    plot_style.apply()
    FIGURES.mkdir(parents=True, exist_ok=True)
    table = collect_rows()
    write_tables(table)
    fig_main(table)
    fig_ablation()


if __name__ == "__main__":
    main()
