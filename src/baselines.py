import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.metrics import classification_report

from data_pipeline import get_grouped_splits

# Format input text
def format_input(data, with_rule=False):
    if with_rule:
        rule = data["Rule-of-Thumb"]
        return "Country: " + data["country_name"] + "\nRule: " + data["Rule-of-Thumb"] + "\nStory: " + data["Story"]
    else:
        return "Country: " + data["country_name"] + "\nStory: " + data["Story"]

# Grid search hyperparameter tuning
def run_grid_search(pipeline, param_grid, X_train, y_train, X_test, y_test, cv):
    grid_search = GridSearchCV(pipeline, param_grid, scoring="macro_f1", cv=cv, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1_score, support = precision_recall_fscore_support(y_test, y_pred, average="macro", zero_division=0)
    report = classification_report(y_test, y_pred, labels=["yes", "no", "neither"], digits=4, zero_division=0)

    print(f"Accuracy: {accuracy:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | Macro F1: {f1_score:.4f} | Support: {support:.4f}")

def run_baselines():
    train_val_data, test_data, cv_folds = get_grouped_splits()

    y_train = train_val_data["gold_label"]
    y_test = test_data["gold_label"]
    
    # Naive: DummyClassifier (most frequent)

    # Course-level baselines
    conditions = [
        ("Without Rule-of-Thumb", False),
        ("With Rule-of-Thumb", True)
    ]

    # TF-IDF + logistic regression for both conditions

    # TF-IDF + Multinomial Naive Bayes for both conditions


if __name__ == "__main__":
    run_baselines()