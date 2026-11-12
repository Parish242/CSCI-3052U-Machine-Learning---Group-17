
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from data_pipeline import get_grouped_splits
from baselines import format_input

SEED = 42
LABELS = ["yes", "no", "neither"]
RESULTS = Path(__file__).resolve().parents[1] / "results"


def make_cv(n_splits, seed=SEED):
    return StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed,
    )


def make_model(model_name):
    if model_name == "Logistic Regression":
        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(max_features=5000)),
            ("lr", LogisticRegression(
                max_iter=1000,
                random_state=SEED,
            )),
        ])
        params = {
            "tfidf__ngram_range": [(1, 1), (1, 2)],
            "lr__C": [0.1, 1.0, 10.0],
        }
    else:
        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(max_features=5000)),
            ("nb", MultinomialNB()),
        ])
        params = {
            "tfidf__ngram_range": [(1, 1), (1, 2)],
            "nb__alpha": [0.1, 0.5, 1.0, 2.0],
        }

    return pipeline, params


def evaluate_protocol(data, group_column, model_name, with_rule):
    X = format_input(data, with_rule=with_rule).reset_index(drop=True)
    y = data["gold_label"].reset_index(drop=True)
    groups = data[group_column].reset_index(drop=True)

    pipeline, params = make_model(model_name)
    outer_cv = make_cv(5)

    rows = []

    for fold, (train_idx, val_idx) in enumerate(
        outer_cv.split(X, y, groups), start=1
    ):
        X_train = X.iloc[train_idx]
        X_val = X.iloc[val_idx]
        y_train = y.iloc[train_idx]
        y_val = y.iloc[val_idx]

        train_groups = groups.iloc[train_idx]
        val_groups = groups.iloc[val_idx]

        # Group separation required by this protocol
        group_overlap = len(
            set(train_groups) & set(val_groups)
        )
        assert group_overlap == 0

        # Independently check both kinds of overlap
        train_rows = data.iloc[train_idx]
        val_rows = data.iloc[val_idx]

        country_overlap = len(
            set(train_rows["Country"]) &
            set(val_rows["Country"])
        )
        background_overlap = len(
            set(train_rows["Background"]) &
            set(val_rows["Background"])
        )

        # Hyperparameter tuning only on outer training rows
        inner_cv = make_cv(3, seed=SEED + fold)

        inner_splits = list(inner_cv.split(
            X_train,
            y_train,
            train_groups,
        ))

        search = GridSearchCV(
            estimator=clone(pipeline),
            param_grid=params,
            scoring="f1_macro",
            cv=inner_splits,
            n_jobs=-1,
            refit=True,
        )

        search.fit(X_train, y_train)
        predictions = search.predict(X_val)

        score = f1_score(
            y_val,
            predictions,
            labels=LABELS,
            average="macro",
            zero_division=0,
        )

        row = {
            "protocol": group_column,
            "model": model_name,
            "condition": (
                "With Rule-of-Thumb"
                if with_rule else "Without Rule-of-Thumb"
            ),
            "fold": fold,
            "train_n": len(train_idx),
            "val_n": len(val_idx),
            "macro_f1": score,
            "inner_cv_f1": search.best_score_,
            "best_params": str(search.best_params_),
            "country_overlap": country_overlap,
            "background_overlap": background_overlap,
        }
        rows.append(row)

        print(
            f"{group_column} | {model_name} | "
            f"RoT={with_rule} | Fold {fold}: "
            f"F1={score:.4f}, "
            f"country overlap={country_overlap}, "
            f"background overlap={background_overlap}"
        )

    return rows


def main():
    train_val, test, _ = get_grouped_splits()

    print(f"Train/validation records: {len(train_val)}")
    print(f"Untouched test records: {len(test)}")

    all_rows = []

    for group_column in ["Background", "Country"]:
        for model_name in [
            "Logistic Regression",
            "Naive Bayes",
        ]:
            for with_rule in [False, True]:
                rows = evaluate_protocol(
                    train_val,
                    group_column,
                    model_name,
                    with_rule,
                )
                all_rows.extend(rows)

    fold_df = pd.DataFrame(all_rows)

    summary = (
        fold_df.groupby(
            ["protocol", "model", "condition"],
            as_index=False,
        )
        .agg(
            mean_macro_f1=("macro_f1", "mean"),
            std_macro_f1=("macro_f1", "std"),
            min_macro_f1=("macro_f1", "min"),
            max_macro_f1=("macro_f1", "max"),
            max_country_overlap=("country_overlap", "max"),
            max_background_overlap=("background_overlap", "max"),
        )
    )

    RESULTS.mkdir(parents=True, exist_ok=True)

    fold_df.to_csv(
        RESULTS / "country_grouped_cv_folds.csv",
        index=False,
    )
    summary.to_csv(
        RESULTS / "country_grouped_cv.csv",
        index=False,
    )

    print("\n=== NESTED CV SUMMARY ===")
    print(summary.to_string(index=False))
    print("\nResults saved to results/")


if __name__ == "__main__":
    main()
