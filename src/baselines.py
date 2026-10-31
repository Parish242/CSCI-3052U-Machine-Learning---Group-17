import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.metrics import classification_report

from data_pipeline import get_grouped_splits

LABELS = ["yes", "no", "neither"]
DUMMY_SEEDS = range(10)
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

# Final "Is what X did (socially) acceptable?" question
QUESTION_PATTERN = r'\s*["“]?\s*Is what [^?]*\?\s*["”]?\s*$'

# Format input text
def format_input(data, with_rule=False, strip_question=False):
    story = data["Story"]
    if strip_question:
        story = story.str.replace(QUESTION_PATTERN, "", regex=True)
    if with_rule:
        return "Country: " + data["country_name"] + "\nRule: " + data["Rule-of-Thumb"] + "\nStory: " + story
    else:
        return "Country: " + data["country_name"] + "\nStory: " + story

# Evaluation over all three labels
def evaluate(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)
    return accuracy, precision, recall, f1

# Grid search hyperparameter tuning
def run_grid_search(pipeline, param_grid, X_train, y_train, X_test, y_test, cv_folds):
    grid_search = GridSearchCV(pipeline, param_grid, scoring="f1_macro", cv=cv_folds, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV Macro F1: {grid_search.best_score_:.4f}")

    # Spread of the best config across the CV folds
    cv_std = grid_search.cv_results_["std_test_score"][grid_search.best_index_]

    # Per-class F1 on each CV fold, averaged over folds
    fold_class_f1 = []
    for train_idx, val_idx in cv_folds:
        fold_model = clone(grid_search.best_estimator_)
        fold_model.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
        fold_pred = fold_model.predict(X_train.iloc[val_idx])
        fold_class_f1.append(f1_score(y_train.iloc[val_idx], fold_pred, labels=LABELS, average=None, zero_division=0))
    cv_class_f1 = np.mean(fold_class_f1, axis=0)

    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    accuracy, precision, recall, f1 = evaluate(y_test, y_pred)

    print(f"Test Accuracy: {accuracy:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | Macro F1: {f1:.4f}")
    print(classification_report(y_test, y_pred, labels=LABELS, digits=4, zero_division=0))

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "macro_f1": f1,
        "cv_macro_f1": grid_search.best_score_,
        "cv_macro_f1_std": cv_std,
        **{f"cv_f1_{label}": f1 for label, f1 in zip(LABELS, cv_class_f1)},
        "best_params": str(grid_search.best_params_),
        "test_samples": len(y_test)
    }

def run_baselines():
    train_val_data, test_data, cv_folds = get_grouped_splits()

    print("\n=== DATA SPLIT VERIFICATION ===")
    print("Training/validation samples:", len(train_val_data))
    print("Test samples:", len(test_data))

    train_groups = set(train_val_data["Background"])
    test_groups = set(test_data["Background"])

    overlap = train_groups.intersection(test_groups)

    print("Overlapping Background groups:", len(overlap))
    print("Training class distribution:")
    print(train_val_data["gold_label"].value_counts())
    print("Test class distribution:")
    print(test_data["gold_label"].value_counts())

    assert len(overlap) == 0, "Data leakage: groups overlap!"

    results = []

    y_train = train_val_data["gold_label"]
    y_test = test_data["gold_label"]

    print("\nNaive baselines")
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(train_val_data, y_train) 
    dummy_pred = dummy.predict(test_data)
    d_accuracy, _, _, d_f1 = evaluate(y_test, dummy_pred)
    print(f"\nDummy classifier (most frequent, always 'yes')")
    print(f"Accuracy: {d_accuracy:.4f} | Macro F1: {d_f1:.4f}")

    results.append({
        "model": "Dummy Classifier",
        "condition": "Without Rule-of-Thumb",
        "accuracy": d_accuracy,
        "macro_f1": d_f1,
        "test_samples": len(y_test)
    })

    # Random guessing in proportion to the training class distribution, averaged over seeds
    strat_scores = []
    for seed in DUMMY_SEEDS:
        strat = DummyClassifier(strategy="stratified", random_state=seed)
        strat.fit(train_val_data, y_train)
        strat_scores.append(evaluate(y_test, strat.predict(test_data)))
    strat_acc = np.array([score[0] for score in strat_scores])
    strat_f1 = np.array([score[3] for score in strat_scores])
    print(f"\nDummy classifier (stratified random, {len(DUMMY_SEEDS)} seeds)")
    print(f"Accuracy: {strat_acc.mean():.4f} | Macro F1: {strat_f1.mean():.4f} +/- {strat_f1.std():.4f}")

    results.append({
        "model": "Dummy Classifier (stratified)",
        "condition": "Without Rule-of-Thumb",
        "accuracy": strat_acc.mean(),
        "macro_f1": strat_f1.mean(),
        "macro_f1_seed_std": strat_f1.std(),
        "test_samples": len(y_test)
    })

    print("\nCourse-level baselines")
    conditions = [
        ("Without Rule-of-Thumb", False),
        ("With Rule-of-Thumb", True)
    ]

    for condition, include_rule in conditions:
        print("\n" + condition)
        X_train_text = format_input(train_val_data, include_rule)
        X_test_text = format_input(test_data, include_rule)

        print("\nTF-IDF + Logistic Regression")
        lr_pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000)),
            ('lr', LogisticRegression(max_iter=1000, random_state=42))
        ])
        lr_param_grid = {
            'tfidf__ngram_range': [(1, 1), (1, 2)],
            'lr__C': [0.1, 1.0, 10.0]
        }
        lr_result = run_grid_search(
            lr_pipeline, lr_param_grid,
            X_train_text, y_train,
            X_test_text, y_test, cv_folds
        )

        lr_result["model"] = "Logistic Regression"
        lr_result["condition"] = condition
        results.append(lr_result)

        print("\nTF-IDF + Multinomial Naive Bayes")
        nb_pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000)),
            ('nb', MultinomialNB())
        ])
        nb_param_grid = {
            'tfidf__ngram_range': [(1, 1), (1, 2)],
            'nb__alpha': [0.1, 0.5, 1.0, 2.0]
        }
        nb_result = run_grid_search(
            nb_pipeline, nb_param_grid,
            X_train_text, y_train,
            X_test_text, y_test, cv_folds
        )

        nb_result["model"] = "Naive Bayes"
        nb_result["condition"] = condition
        results.append(nb_result)

    RESULTS_DIR.mkdir(exist_ok=True)

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        RESULTS_DIR / "baseline_results.csv",
        index=False
    )

    print("\n=== BASELINE RESULTS SAVED ===")
    print(results_df.to_string(index=False))
    print("Saved to results/baseline_results.csv")

# Same grid searches with the final question removed from every story
def run_question_ablation():
    train_val_data, test_data, cv_folds = get_grouped_splits()

    y_train = train_val_data["gold_label"]
    y_test = test_data["gold_label"]

    kept = format_input(train_val_data, strip_question=True).str.contains(r"\?\s*$").sum()
    print(f"\nQuestion ablation: {kept} train/val stories still end with a question")

    results = []
    conditions = [
        ("Without Rule-of-Thumb", False),
        ("With Rule-of-Thumb", True)
    ]

    for condition, include_rule in conditions:
        print("\n" + condition + " (question removed)")
        X_train_text = format_input(train_val_data, include_rule, strip_question=True)
        X_test_text = format_input(test_data, include_rule, strip_question=True)

        print("\nTF-IDF + Logistic Regression")
        lr_pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000)),
            ('lr', LogisticRegression(max_iter=1000, random_state=42))
        ])
        lr_param_grid = {
            'tfidf__ngram_range': [(1, 1), (1, 2)],
            'lr__C': [0.1, 1.0, 10.0]
        }
        lr_result = run_grid_search(
            lr_pipeline, lr_param_grid,
            X_train_text, y_train,
            X_test_text, y_test, cv_folds
        )

        lr_result["model"] = "Logistic Regression"
        lr_result["condition"] = condition
        results.append(lr_result)

        print("\nTF-IDF + Multinomial Naive Bayes")
        nb_pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000)),
            ('nb', MultinomialNB())
        ])
        nb_param_grid = {
            'tfidf__ngram_range': [(1, 1), (1, 2)],
            'nb__alpha': [0.1, 0.5, 1.0, 2.0]
        }
        nb_result = run_grid_search(
            nb_pipeline, nb_param_grid,
            X_train_text, y_train,
            X_test_text, y_test, cv_folds
        )

        nb_result["model"] = "Naive Bayes"
        nb_result["condition"] = condition
        results.append(nb_result)

    RESULTS_DIR.mkdir(exist_ok=True)

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        RESULTS_DIR / "baseline_results_no_question.csv",
        index=False
    )

    print("\n=== QUESTION ABLATION RESULTS SAVED ===")
    print(results_df.to_string(index=False))
    print("Saved to results/baseline_results_no_question.csv")

if __name__ == "__main__":
    run_baselines()
    run_question_ablation()