from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.metrics import classification_report

from data_pipeline import get_grouped_splits

LABELS = ["yes", "no", "neither"]
DUMMY_SEEDS = range(10)

# Format input text
def format_input(data, with_rule=False):
    if with_rule:
        return "Country: " + data["country_name"] + "\nRule: " + data["Rule-of-Thumb"] + "\nStory: " + data["Story"]
    else:
        return "Country: " + data["country_name"] + "\nStory: " + data["Story"]

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

    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    accuracy, precision, recall, f1 = evaluate(y_test, y_pred)

    print(f"Test Accuracy: {accuracy:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | Macro F1: {f1:.4f}")
    print(classification_report(y_test, y_pred, labels=LABELS, digits=4, zero_division=0))

def run_baselines():
    train_val_data, test_data, cv_folds = get_grouped_splits()

    y_train = train_val_data["gold_label"]
    y_test = test_data["gold_label"]

    print("\nNaive baselines")
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(train_val_data, y_train) 
    dummy_pred = dummy.predict(test_data)
    d_accuracy, _, _, d_f1 = evaluate(y_test, dummy_pred)
    print(f"\nDummy classifier (most frequent, always 'yes')")
    print(f"Accuracy: {d_accuracy:.4f} | Macro F1: {d_f1:.4f}")

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
        run_grid_search(lr_pipeline, lr_param_grid, X_train_text, y_train, X_test_text, y_test, cv_folds)

        print("\nTF-IDF + Multinomial Naive Bayes")
        nb_pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000)),
            ('nb', MultinomialNB())
        ])
        nb_param_grid = {
            'tfidf__ngram_range': [(1, 1), (1, 2)],
            'nb__alpha': [0.1, 0.5, 1.0, 2.0]
        }
        run_grid_search(nb_pipeline, nb_param_grid, X_train_text, y_train, X_test_text, y_test, cv_folds)

if __name__ == "__main__":
    run_baselines()