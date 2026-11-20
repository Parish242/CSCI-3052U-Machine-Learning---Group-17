"""Milestone 5: error patterns on the shared NormAd test split. Run from any directory."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupShuffleSplit

from paired_bootstrap import LLM_FILES

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = RESULTS / "error_analysis"
LABELS = ["yes", "no", "neither"]
PRED = LABELS + ["unparseable"]
KEY = ["model", "rule", "question"]
MIN_STORIES = 10  # only for ranking rates; the breakdown CSVs include every group
CLASSICAL = "results/classical_test_predictions.csv"
RULE_NAMES = {"With Rule-of-Thumb": "with", "Without Rule-of-Thumb": "without"}

def labels(values):
    return values.astype(str).str.strip().str.lower().replace({"neutral": "neither"})


def load_runs():
    data = pd.read_csv(ROOT / "data/normad_clean.csv", dtype={"ID": str})
    data = data.rename(columns={"Gold Label": "gold", "Country_Display": "country",
                                "Subaxis": "subaxis", "Background": "background"})
    data["gold"] = labels(data["gold"])
    if not data.ID.is_unique or not data.gold.isin(LABELS).all():
        raise ValueError("Duplicate story IDs or unexpected gold labels in NormAd")
    split = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_idx, test_idx = next(split.split(data, groups=data.background))
    train, test = data.iloc[train_idx], data.iloc[test_idx].set_index("ID")
    if len(test) != 521 or set(test.background) & set(train.background):
        raise ValueError("Expected the project's 521-story, background-disjoint test split")
    if test[["country", "subaxis", "background"]].isna().any().any():
        raise ValueError("Missing group metadata on the test split")
    runs = []

    def add(preds, model, rule, question, source, seed=0, full=False):
        if not preds.ID.is_unique:
            raise ValueError(f"{source}: repeated IDs in {model}/{rule}/{question}")
        p = preds.set_index("ID")
        expected_ids = data.ID if full else test.index
        if set(p.index) != set(expected_ids):
            raise ValueError(f"{source}: IDs do not match the expected split")
        p = p.loc[test.index]  # join on ID, never on CSV row position
        if not labels(p.gold).equals(test.gold):
            raise ValueError(f"{source}: prediction and dataset gold labels disagree")
        pred = labels(p.pred)
        block = test[["gold", "country", "subaxis", "background"]].copy()
        block["pred"] = pred.where(pred.isin(LABELS), "unparseable")
        block["raw_output"] = p["raw_output"] if "raw_output" in p else ""
        runs.append(block.assign(model=model, rule=rule, question=question,
                                 seed=seed, source_file=source).reset_index())

    classical = pd.read_csv(ROOT / CLASSICAL, dtype={"ID": str}, keep_default_na=False)
    for (model, condition, question), p in classical.groupby(["model", "condition", "question"]):
        add(p, model, RULE_NAMES[condition], question, CLASSICAL)

    for run, filename in LLM_FILES.items():  # same six full exports used for the paired bootstrap
        model, condition, *ablation = run.split(" | ")
        p = pd.read_csv(RESULTS / filename, dtype={"ID": str}, keep_default_na=False)
        add(p, model, {"With RoT": "with", "Without RoT": "without"}[condition],
            "removed" if ablation else "kept", "results/" + filename, full=True)

    # baselines.py saves only aggregate dummy scores. Replay its strategies and seeds
    # in the original split order, which matters for matching stratified draws to IDs.
    for strategy, seeds in [("most_frequent", [0]), ("stratified", range(10))]:
        for seed in seeds:
            clf = DummyClassifier(strategy=strategy, random_state=seed)
            clf.fit(train[["country"]], train.gold)
            pred = clf.predict(test[["country"]])
            p = pd.DataFrame({"ID": test.index, "gold": test.gold.to_numpy(), "pred": pred})
            add(p, "Dummy " + strategy, "n/a", "n/a", "src/baselines.py (replayed)", seed)
    scored = pd.concat(runs, ignore_index=True)
    return scored.assign(wrong=scored.gold != scored.pred, invalid=scored.pred == "unparseable")


def breakdown(scored, by=()):
    out = scored.groupby(KEY + list(by), as_index=False).agg(
        n_stories=("ID", "nunique"), n_backgrounds=("background", "nunique"),
        n_predictions=("ID", "size"), errors=("wrong", "sum"), unparseable=("invalid", "sum"))
    out["error_rate"] = out.errors / out.n_predictions
    return out


def model_summary(scored):
    summary = breakdown(scored)
    scores = []
    for key, g in scored.groupby(KEY):
        f1 = [f1_score(rep.gold, rep.pred, labels=LABELS, average="macro", zero_division=0)
              for _, rep in g.groupby("seed")]
        scores.append((*key, len(f1), np.mean(f1)))
    summary["accuracy"] = 1 - summary.error_rate
    return summary.merge(pd.DataFrame(scores, columns=KEY + ["n_runs", "macro_f1"]), on=KEY)


def confusion_matrices(scored):
    rows = []
    for key, g in scored.groupby(KEY):
        for gold in LABELS:
            sub = g[g.gold == gold]
            counts = sub.pred.value_counts().reindex(PRED, fill_value=0)
            rows.append((*key, gold, sub.ID.nunique(), len(sub), *counts))
    return pd.DataFrame(rows, columns=KEY + ["gold", "n_stories", "n_predictions"] +
                        ["pred_" + label for label in PRED])


def llm_hotspots(countries, subaxes):
    rows = []
    for axis, table in [("country", countries), ("subaxis", subaxes)]:
        for key, g in table[table.model.isin(["Gemma", "InternLM"])].groupby(KEY):
            eligible = g[g.n_stories >= MIN_STORIES]
            leaders = [("most_errors", g[g.errors == g.errors.max()]),
                       ("highest_rate", eligible[eligible.error_rate == eligible.error_rate.max()])]
            for measure, top in leaders:
                for _, r in top.iterrows():  # keep all ties, including countries of different sizes
                    rows.append((*key, axis, measure, r[axis], r.n_stories, r.n_backgrounds,
                                 r.errors, r.error_rate))
    return pd.DataFrame(rows, columns=KEY + ["group_type", "ranked_by", "group", "n_stories",
                                             "n_backgrounds", "errors", "error_rate"])


# These cases illustrate distinct failure modes; they are not a random error sample.
# ID, model, rule, question, expected gold and prediction, observation, possible explanation.
CASES = [
    ("1403", "InternLM", "without", "kept", "no", "neither",
     "A dinner guest starts eating as soon as food is served.",
     "The rule says to wait for the group and the blessing. Without it the timing may seem unremarkable; with the rule, InternLM answers no."),
    ("951", "InternLM", "with", "kept", "no", "neither",
     "Ben is pleased with his gift and unwraps it on the spot.",
     "The happy reaction sounds polite, but the supplied rule specifically says to wait before opening the gift."),
    ("528", "Gemma", "without", "kept", "yes", "neither",
     "A dinner guest takes a portion she can finish, then clears her plate.",
     "Without the rule this can read as a personal choice. Once given the rule about finishing the meal, Gemma answers yes."),
    ("1857", "Gemma", "with", "kept", "neither", "no",
     "A guest stays standing after being offered a seat; the supplied rule is about masks when ill.",
     "The rule cannot settle the seating question. The negative answer may judge the guest's manners instead; without the rule Gemma answers neither."),
    ("889", "Gemma", "with", "kept", "yes", "unparseable",
     "Mike returns a scarf using both hands, which follows the supplied rule.",
     "The saved output stops at \"Mike's action is\" and gives no label. This is a response-format failure; the intended label is unknown."),
    ("1607", "Naive Bayes", "with", "kept", "no", "yes",
     "A guest returns a gift bag without putting anything inside.",
     "The return itself matches much of the rule's wording, but leaving the bag empty violates it. A word-based classifier can miss that negation."),
    ("1254", "Logistic Regression", "without", "removed", "no", "yes",
     "Anna arrives at 10:30 for a 10:00 meeting.",
     "Recognizing the half-hour delay requires relating the two times. The no-rule model gets this case right when the original question wording is kept."),
    ("1609", "Logistic Regression", "without", "kept", "yes", "no",
     "Sarah waits until later to correct a colleague privately.",
     "The CSV gold is yes, but the dataset explanation ends by calling her action unacceptable. This is a scored error with conflicting annotation evidence, not a clean model failure."),
]


def error_examples(scored):
    rows = []
    for id_, model, rule, question, gold, pred, scene, reason in CASES:
        e = scored[(scored.ID == id_) & (scored.model == model) & (scored.rule == rule) &
                   (scored.question == question)]
        if len(e) != 1 or (e.iloc[0].gold, e.iloc[0].pred) != (gold, pred) or gold == pred:
            raise ValueError(f"Example {id_}/{model} is no longer the documented error")
        r = e.iloc[0]
        rows.append((id_, model, rule, question, r.country, r.subaxis, gold, pred,
                     r.raw_output, scene, reason, r.source_file))
    return pd.DataFrame(rows, columns=["ID"] + KEY + ["country", "subaxis", "gold", "pred",
                        "raw_output", "story_summary", "possible_reason", "source_file"])


def plot_matrices(cm, configs, title, path):
    # Color is the percentage within the gold row. The text is the actual count.
    nrows = len(configs) // 2
    fig, axes = plt.subplots(nrows, 2, figsize=(9.5, 2.25 * nrows + 0.9))
    colors = plt.get_cmap("Blues")
    indexed = cm.set_index(KEY)
    for ax, key in zip(axes.flat, configs):
        matrix = indexed.loc[[key]].set_index("gold").loc[LABELS, ["pred_" + p for p in PRED]]
        counts = matrix.to_numpy()
        shares = counts / counts.sum(axis=1, keepdims=True)
        ax.pcolormesh(shares, cmap=colors, vmin=0, vmax=1, edgecolors="white", linewidth=1)
        for i, j in np.ndindex(counts.shape):
            ax.text(j + .5, i + .5, str(counts[i, j]), ha="center", va="center", fontsize=10,
                    color="white" if shares[i, j] > .55 else "#162737")
        model, rule, question = key
        if model.startswith("Dummy"):
            panel = "Most frequent dummy" if model.endswith("frequent") else "Stratified dummy (10 seeds pooled)"
        else:
            panel = f"{model} · {rule} rule" + (" · question removed" if question == "removed" else "")
        ax.set_title(panel, fontsize=10, loc="left")
        ax.set_xticks(np.arange(4) + .5, ["yes", "no", "neither", "unparseable"], fontsize=8)
        ax.set_yticks(np.arange(3) + .5, LABELS, fontsize=8)
        ax.invert_yaxis()
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.suptitle(title, x=.09, y=.99, ha="left", fontsize=15, fontweight="bold")
    fig.text(.09, 1 - .52 / fig.get_figheight(),
             "Rows: gold label   |   Columns: prediction   |   Darker blue: larger fraction of that gold row",
             fontsize=9)
    fig.subplots_adjust(left=.09, right=.985, bottom=.035, top=1 - 1.03 / fig.get_figheight(),
                        wspace=.24, hspace=.57)
    with matplotlib.rc_context({"svg.fonttype": "none", "svg.hashsalt": "normad-m5"}):
        fig.savefig(path, metadata={"Date": None})
    plt.close(fig)
    # Matplotlib leaves spaces at SVG path line breaks; keep the generated Git diff clean.
    svg = path.read_text(encoding="utf-8")
    path.write_text("\n".join(line.rstrip() for line in svg.splitlines()) + "\n", encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    scored = load_runs()
    countries = breakdown(scored, ["country"])
    subaxes = breakdown(scored, ["subaxis"])
    cm = confusion_matrices(scored)
    outputs = {"model_summary": model_summary(scored), "confusion_matrices": cm,
               "by_country": countries, "by_subaxis": subaxes,
               "llm_hotspots": llm_hotspots(countries, subaxes), "example_errors": error_examples(scored)}
    for name, table in outputs.items():
        table.to_csv(OUT / f"{name}.csv", index=False, float_format="%.6f")
    baselines = [(m, r, q) for q in ("kept", "removed")
                 for m in ("Logistic Regression", "Naive Bayes") for r in ("without", "with")]
    baselines += [("Dummy most_frequent", "n/a", "n/a"), ("Dummy stratified", "n/a", "n/a")]
    llms = [(m, r, q) for m, q in (("InternLM", "kept"), ("Gemma", "kept"), ("InternLM", "removed"))
            for r in ("without", "with")]
    plot_matrices(cm, baselines, "Classical and dummy baselines", OUT / "confusion_baselines.svg")
    plot_matrices(cm, llms, "Language models", OUT / "confusion_llms.svg")
    print(f"{scored.ID.nunique()} test stories; {len(outputs['model_summary'])} configurations; "
          f"{len(outputs['example_errors'])} checked examples. Results: {OUT}")


if __name__ == "__main__":
    main()
