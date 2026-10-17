# CSCI-3052U Machine Learning — Group 17

Evaluating LLMs and classical baselines on the [NormAd](https://huggingface.co/datasets/akhilayerukola/NormAd) cultural-norm dataset.

## Repository layout

```
data/                    NormAd CSVs (raw, clean, preprocessed, 50-row trial subset)
notebooks/               Exploratory data analysis
src/
  data_pipeline.py       Load NormAd and build the train/val/test split
  baselines.py           TF-IDF / classical baselines + grid search
results/
  <model>/<run>/         LLM outputs per model; run = trial (50 rows) or full
    *_with-rule*.csv       predictions when the rule-of-thumb is in the prompt
    *_without-rule*.csv    predictions without it
    *_report.txt           sklearn classification reports
    *_run-notes*           model, quantization, hardware, issues
requirements.txt
```

## Setup

```
python -m venv .venv
.venv\Scripts\activate        # Windows; use `source .venv/bin/activate` elsewhere
pip install -r requirements.txt
```

## Classical Baseline Evaluation

This project implements three classical machine learning baselines for the NormAd-ETI dataset:

1. **DummyClassifier** — Predicts the most frequent class as a naive benchmark.
2. **TF-IDF + Logistic Regression** — Uses TF-IDF text features with a logistic regression classifier.
3. **TF-IDF + Multinomial Naive Bayes** — Uses TF-IDF text features with a probabilistic classifier.

Logistic Regression and Naive Bayes are evaluated under two conditions:
- **Without Rule-of-Thumb:** Country + Story
- **With Rule-of-Thumb:** Country + Rule-of-Thumb + Story

### Data Splitting and Evaluation

The dataset contains 2,615 examples across three classes: `yes`, `no`, and `neither`.

- Training/validation set: 2,094 examples
- Held-out test set: 521 examples
- Splitting method: `GroupShuffleSplit`, grouped by `Background`
- Cross-validation: 5-fold `StratifiedGroupKFold`
- Random seed: 42
- Primary metric: Macro-F1
- Additional metrics: Accuracy, precision, and recall

Data split verification confirmed **zero overlapping Background groups** between the training/validation and test sets.

Hyperparameters were selected using `GridSearchCV` with `f1_macro` scoring. TF-IDF vectorization was included inside the scikit-learn pipelines to prevent vocabulary leakage during cross-validation.

### Running the Baselines

After installing the dependencies from `requirements.txt`, run the following command from the repository root:

```bash
python src/baselines.py
```

The script trains and evaluates all five baseline configurations, prints classification reports, and exports the results to:

```text
results/baseline_results.csv
```

### Baseline Results

All results below were evaluated on the same 521-example held-out test set.

| Model | Rule-of-Thumb | Accuracy | Macro-F1 |
|---|---|---:|---:|
| DummyClassifier | N/A | 0.3647 | 0.1782 |
| TF-IDF + Logistic Regression | Without | **0.9194** | **0.9191** |
| TF-IDF + Logistic Regression | With | 0.9079 | 0.9078 |
| TF-IDF + Multinomial Naive Bayes | Without | 0.8100 | 0.8083 |
| TF-IDF + Multinomial Naive Bayes | With | 0.7908 | 0.7895 |

### Results Interpretation

TF-IDF + Logistic Regression without rule-of-thumb information achieved the highest performance, with 91.94% accuracy and a macro-F1 of 0.9191.

Multinomial Naive Bayes without rule-of-thumb information achieved a macro-F1 of 0.8083, while the majority-class DummyClassifier achieved 0.1782.

Including rule-of-thumb information slightly reduced performance for both classical models:

- Logistic Regression: 0.9191 to 0.9078 (decrease of 0.0113)
- Naive Bayes: 0.8083 to 0.7895 (decrease of 0.0188)

These findings suggest that adding rule-of-thumb information did not improve the classical TF-IDF baselines under the current experimental setup.

The high classification scores warrant further investigation into possible dataset shortcuts or shared linguistic patterns. Comparisons with the existing LLM replication results must also account for differences in evaluation subsets and protocols.

### Reproducibility

The classical baselines were executed using the following environment:

| Software | Version |
|---|---|
| Python | 3.12.0 |
| scikit-learn | 1.9.1 |
| pandas | 3.0.6 |
| Hugging Face Datasets | 5.0.1 |

Dependencies are specified in `requirements.txt`.

The dataset is loaded from Hugging Face:
[akhilayerukola/NormAd](https://huggingface.co/datasets/akhilayerukola/NormAd).

The random seed, splitting strategy, cross-validation procedure, and hyperparameter search spaces are defined in `src/data_pipeline.py` and `src/baselines.py`.

The exact dataset revision should be recorded to ensure future reproducibility.
