import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.dummy import DummyClassifier
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

# Evaluation
def evaluate_model(model, name, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f_score = precision_recall_fscore_support(y_test, y_pred, average='macro', zero_division=0)
    report = classification_report(y_test, y_pred, labels=["yes", "no", "neither"], digits=4, zero_division=0)
