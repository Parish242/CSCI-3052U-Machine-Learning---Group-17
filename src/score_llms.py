import pandas as pd
from pathlib import Path

from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix

from data_pipeline import get_grouped_splits

LABELS = ["yes", "no", "neither"]
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

# Full-run prediction files (2,615 items each)
LLM_RUNS = [
    ("internlm2.5-7b", "Without Rule-of-Thumb", "internlm2.5-7b/full/internlm2.5-7b_without-rule.csv"),
    ("internlm2.5-7b", "With Rule-of-Thumb", "internlm2.5-7b/full/internlm2.5-7b_with-rule.csv"),
    ("gemma-2-27b", "Without Rule-of-Thumb", "gemma-2-27b/full/gemma-2-27b_full_without_rule.csv"),
    ("gemma-2-27b", "With Rule-of-Thumb", "gemma-2-27b/full/gemma-2-27b_full_with_rule.csv"),
]

# Load predictions and map neutral -> neither; anything else outside LABELS counts as unparseable
def load_predictions(path):
    preds = pd.read_csv(RESULTS_DIR / path)
    for col in ["gold", "pred"]:
        preds[col] = preds[col].astype(str).str.strip().str.lower().replace({"neutral": "neither"})
    return preds

# Score predictions on a subset of IDs; unparseable outputs count as errors
def score(preds, ids):
    subset = preds[preds["ID"].isin(ids)]
    y_true, y_pred = subset["gold"], subset["pred"]
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)
    class_f1 = f1_score(y_true, y_pred, labels=LABELS, average=None, zero_division=0)
    return {
        "n": len(subset),
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1,
        "precision": precision,
        "recall": recall,
        **{f"f1_{label}": value for label, value in zip(LABELS, class_f1)},
        "unparseable": int((~y_pred.isin(LABELS)).sum()),
    }, confusion_matrix(y_true, y_pred, labels=LABELS)

def score_llms():
    train_val_data, test_data, cv_folds = get_grouped_splits()

    test_rows, cv_rows = [], []
    for model, condition, path in LLM_RUNS:
        preds = load_predictions(path)

        # Held-out test set, same 521 items as the baselines
        test_scores, cm = score(preds, test_data["ID"])
        test_rows.append({"model": model, "condition": condition, **test_scores})
        print(f"\n{model} | {condition} | test macro F1: {test_scores['macro_f1']:.4f}")
        print("Confusion matrix (rows = gold, cols = predicted):", LABELS)
        print(cm)

        # Validation part of each CV fold, same folds as the baseline grid search
        for fold, (_, val_idx) in enumerate(cv_folds):
            fold_scores, _ = score(preds, train_val_data["ID"].iloc[val_idx])
            cv_rows.append({
                "model": model, "condition": condition, "fold": fold,
                **{k: fold_scores[k] for k in ["n", "macro_f1", "f1_yes", "f1_no", "f1_neither", "unparseable"]}
            })

    test_df = pd.DataFrame(test_rows)
    cv_df = pd.DataFrame(cv_rows)
    test_df.to_csv(RESULTS_DIR / "llm_test_scores.csv", index=False)
    cv_df.to_csv(RESULTS_DIR / "llm_cv_scores.csv", index=False)

    print("\n=== LLM SCORES SAVED ===")
    print(test_df.to_string(index=False))
    print(cv_df.groupby(["model", "condition"])["macro_f1"].agg(["mean", "std"]))
    print("Saved to results/llm_test_scores.csv and results/llm_cv_scores.csv")

if __name__ == "__main__":
    score_llms()
