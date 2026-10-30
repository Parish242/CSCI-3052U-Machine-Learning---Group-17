# CSCI-3052U Machine Learning — Group 17

Evaluating LLMs and classical baselines on the [NormAd](https://huggingface.co/datasets/akhilayerukola/NormAd) cultural-norm dataset.

## Repository layout

```
data/                    NormAd CSVs (raw, clean, preprocessed, 50-row trial subset)
notebooks/               Exploratory data analysis
src/
  data_pipeline.py       Load NormAd (pinned revision) and build the train/val/test split
  baselines.py           Naive and TF-IDF baselines + grid search + question ablation
  score_llms.py          Score LLM predictions on the baseline test set and CV folds
results/
  baseline_results.csv   Baseline table (written by src/baselines.py)
  baseline_results_no_question.csv
                         TF-IDF baselines with the final question removed (src/baselines.py)
  llm_test_scores.csv    LLM scores on the 521-item test set (src/score_llms.py)
  llm_cv_scores.csv      LLM scores per CV fold (src/score_llms.py)
  <model>/<run>/         LLM outputs per model (not part of the baseline milestone);
                         run = trial (50 rows) or full; each holds predictions with and
                         without the rule-of-thumb, classification reports, and run notes
requirements.txt
```

## Setup

```
python -m venv .venv
.venv\Scripts\activate        # Windows; use `source .venv/bin/activate` elsewhere
pip install -r requirements.txt
```

## Classical Baseline Evaluation

This milestone implements two naive benchmarks and two classical machine learning baselines for the NormAd-ETI dataset:

1. **DummyClassifier (most frequent)** — Always predicts the majority class (`yes`).
2. **DummyClassifier (stratified)** — Guesses at random in proportion to the training class distribution, averaged over 10 seeds. This is the more meaningful macro-F1 floor, since always-`yes` scores zero on two of three classes.
3. **TF-IDF + Logistic Regression** — Uses TF-IDF text features with a logistic regression classifier.
4. **TF-IDF + Multinomial Naive Bayes** — Uses TF-IDF text features with a probabilistic classifier.

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

After installing the dependencies from requirements.txt, run the following command from the repository root:

```bash
python src/baselines.py
```

The script trains and evaluates all five baseline configurations, prints classification reports, and exports the results to:

```text
results/baseline_results.csv
```

### Baseline Results

Test columns are on the same 521-example held-out test set. CV macro-F1 is the mean ± standard deviation over the 5 training folds for the selected hyperparameters.

| Model | Rule-of-Thumb | CV Macro-F1 | Test Accuracy | Test Macro-F1 |
|---|---|---:|---:|---:|
| DummyClassifier (most frequent) | N/A | — | 0.3647 | 0.1782 |
| DummyClassifier (stratified, 10 seeds) | N/A | — | 0.3326 | 0.3299 ± 0.0148 |
| TF-IDF + Logistic Regression | Without | 0.9010 ± 0.0123 | **0.9194** | **0.9191** |
| TF-IDF + Logistic Regression | With | 0.8991 ± 0.0177 | 0.9079 | 0.9078 |
| TF-IDF + Multinomial Naive Bayes | Without | 0.8131 ± 0.0247 | 0.8100 | 0.8083 |
| TF-IDF + Multinomial Naive Bayes | With | 0.7835 ± 0.0232 | 0.7908 | 0.7895 |

Selected hyperparameters: Logistic Regression `C=10`, bigrams in both conditions; Naive Bayes `alpha=0.5` (without) / `1.0` (with), bigrams.

### Results Interpretation

TF-IDF + Logistic Regression without rule-of-thumb information achieved the highest performance, with 91.94% accuracy and a macro-F1 of 0.9191.

Multinomial Naive Bayes without rule-of-thumb information achieved a macro-F1 of 0.8083, compared with 0.1782 for the majority-class DummyClassifier and 0.3299 for stratified random guessing.

Every test score is within 1.5 CV standard deviations of its CV estimate, so the held-out split behaves like the training folds and there is no sign of overfitting during hyperparameter selection.

Including rule-of-thumb information slightly reduced performance for both classical models:

- Logistic Regression: 0.9191 to 0.9078 (decrease of 0.0113)
- Naive Bayes: 0.8083 to 0.7895 (decrease of 0.0188)

These findings suggest that adding rule-of-thumb information did not improve the classical TF-IDF baselines under the current experimental setup.

### Exploratory Question Ablation (for Milestone 4)

This ablation covers the TF-IDF baselines only. Rerunning both LLMs with the question removed or fixed to one wording is planned for Milestone 5.

| Model | Rule-of-Thumb | CV Macro-F1 | Test Accuracy | Test Macro-F1 |
|---|---|---:|---:|---:|
| TF-IDF + Logistic Regression | Without | 0.7978 ± 0.0192 | 0.7908 | 0.7887 |
| TF-IDF + Logistic Regression | With | 0.7974 ± 0.0155 | 0.8081 | 0.8086 |
| TF-IDF + Multinomial Naive Bayes | Without | 0.7626 ± 0.0240 | 0.7639 | 0.7613 |
| TF-IDF + Multinomial Naive Bayes | With | 0.7278 ± 0.0158 | 0.7409 | 0.7393 |

Removing the question lowers Logistic Regression from 0.9191 to 0.7887 macro-F1, so part of the baseline score comes from the question template. With the question removed, the rule-of-thumb now helps Logistic Regression (0.7887 to 0.8086).

### LLM Scores on the Baseline Split

`src/score_llms.py` scores the full-run LLM predictions in `results/<model>/full/` on the same 521-item test set and 5 CV folds as the baselines (`neutral` is mapped to `neither`; unparseable outputs count as errors):

```bash
python src/score_llms.py
```

| Model | Rule-of-Thumb | CV Macro-F1 | Test Accuracy | Test Macro-F1 |
|---|---|---:|---:|---:|
| InternLM2.5-7B-chat (8-bit) | Without | 0.3869 ± 0.0207 | 0.4050 | 0.3543 |
| InternLM2.5-7B-chat (8-bit) | With | 0.6086 ± 0.0368 | 0.6507 | 0.6273 |
| Gemma-2-27B-it (4-bit) | Without | 0.4399 ± 0.0307 | 0.4319 | 0.4218 |
| Gemma-2-27B-it (4-bit) | With | 0.5708 ± 0.0211 | 0.5374 | 0.5470 |

### Reproducibility

The classical baselines were executed using the following environment:

| Software | Version |
|---|---|
| Python | 3.14.3 |
| scikit-learn | 1.9.1 |
| pandas | 3.0.6 |
| Hugging Face Datasets | 5.0.1 |

Dependencies are specified in `requirements.txt`.

The dataset is loaded from Hugging Face:
[akhilayerukola/NormAd](https://huggingface.co/datasets/akhilayerukola/NormAd), pinned to revision `bb1a2997b6ab74b1f901493b15a67c2250a366c9` in `src/data_pipeline.py`. Rerunning `src/baselines.py` with this revision reproduces every value in `results/baseline_results.csv` exactly.

The random seed, splitting strategy, cross-validation procedure, and hyperparameter search spaces are defined in `src/data_pipeline.py` and `src/baselines.py`.
