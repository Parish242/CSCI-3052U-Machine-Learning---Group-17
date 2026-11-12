
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from data_pipeline import get_grouped_splits
from baselines import format_input

RESULTS = Path(__file__).resolve().parents[1] / "results"

CONFIGS = [
    ("Logistic Regression", False, 10.0),
    ("Logistic Regression", True, 10.0),
    ("Naive Bayes", False, 0.5),
    ("Naive Bayes", True, 1.0),
]


def main():
    train, test, _ = get_grouped_splits()
    outputs = []

    for name, with_rule, param in CONFIGS:
        if name == "Logistic Regression":
            clf = LogisticRegression(
                C=param, max_iter=1000, random_state=42
            )
        else:
            clf = MultinomialNB(alpha=param)

        model = Pipeline([
            ("tfidf", TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 2),
            )),
            ("clf", clf),
        ])

        model.fit(
            format_input(train, with_rule=with_rule),
            train["gold_label"],
        )

        predictions = model.predict(
            format_input(test, with_rule=with_rule)
        )

        output = pd.DataFrame({
            "ID": test["ID"],
            "gold": test["gold_label"],
            "pred": predictions,
            "model": name,
            "condition": (
                "With Rule-of-Thumb"
                if with_rule else "Without Rule-of-Thumb"
            ),
        })
        outputs.append(output)

    result = pd.concat(outputs, ignore_index=True)

    RESULTS.mkdir(exist_ok=True)
    result.to_csv(
        RESULTS / "classical_test_predictions.csv",
        index=False,
    )

    print(result.groupby(
        ["model", "condition"]
    ).size())
    print("Saved classical_test_predictions.csv")


if __name__ == "__main__":
    main()
