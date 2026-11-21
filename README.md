# CSCI-3052U Machine Learning — Group 17

Evaluating LLMs and classical baselines on the [NormAd](https://huggingface.co/datasets/akhilayerukola/NormAd) cultural-norm dataset.

## Repository layout

```
data/                    NormAd CSVs (raw, clean, preprocessed, 50-row trial subset)
notebooks/
  normad_eda.ipynb       Exploratory data analysis
  normad_internlm2_5_7b_inference.ipynb
                         InternLM-2.5-7B inference on NormAd (Colab T4)
  Gemma-27B.ipynb        Gemma-2-27B inference, v2: paper prompts, same as InternLM (Kaggle, 2 GPUs; not run yet)
  Gemma-27B_v1-prompt.ipynb
                         Gemma-2-27B v1 prompt: closest code to the reported Gemma results (see its header)
src/
  data_pipeline.py       Load NormAd (pinned revision) and build the train/val/test split
  baselines.py           Naive and TF-IDF baselines + grid search + question ablation
  score_llms.py          Score LLM predictions on the baseline test set and CV folds
  export_test_predictions.py
                         Per-item test predictions of the tuned classical models
  paired_bootstrap.py    95% bootstrap CIs per model and paired differences between models
  cluster_bootstrap_check.py
                         Same differences, resampling whole Background passages instead of items
  country_grouped_cv.py  Robustness: CV grouped by country (unseen countries) vs by Background
  make_results.py        Main results table and Figures 1-2
  error_analysis.py      Confusion matrices, per-country/subaxis errors, eight documented error cases
  plot_style.py          Shared figure colors and style
results/
  baseline_results.csv   Baseline table (written by src/baselines.py)
  baseline_results_no_question.csv
                         TF-IDF baselines with the final question removed (src/baselines.py)
  tuning/                Every grid-search configuration and its per-fold CV scores (src/baselines.py)
  llm_test_scores.csv    LLM scores on the 521-item test set (src/score_llms.py)
  llm_cv_scores.csv      LLM scores per CV fold (src/score_llms.py)
  classical_test_predictions.csv
                         Per-item classical test predictions (src/export_test_predictions.py)
  bootstrap_ci.csv       95% test CIs per model (src/paired_bootstrap.py)
  bootstrap_differences.csv
                         Paired differences between models with 95% CIs (src/paired_bootstrap.py)
  cluster_bootstrap_check.csv
                         Item vs Background-cluster bootstrap CIs (src/cluster_bootstrap_check.py)
  country_grouped_cv.csv, country_grouped_cv_folds.csv
                         Country- vs Background-grouped CV, summary and per fold (src/country_grouped_cv.py)
  main_table.csv, main_table.md
                         Main results table (src/make_results.py)
  figures/               Figures 1-2 (src/make_results.py)
  error_analysis/        Error-analysis tables, confusion-matrix SVGs, and write-up (src/error_analysis.py)
  <model>/<run>/         LLM outputs per model; run = trial (50 rows) or full (includes question-ablation
                         runs); each holds predictions with and without the rule-of-thumb, classification
                         reports, and run notes
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

### LLM Scores on the Baseline Split

`src/score_llms.py` scores the full-run LLM predictions in `results/<model>/full/` on the same 521-item test set and 5 CV folds as the baselines (`neutral` is mapped to `neither`; unparseable outputs count as errors):

```bash
python src/score_llms.py
```

**LLM question ablation.** Set `STRIP_QUESTION = True` in the InternLM notebook to strip the final question (same regex as the `baselines.py` ablation).

| Model | Rule-of-Thumb | Question | CV Macro-F1 | Test Accuracy | Test Macro-F1 |
|---|---|---|---:|---:|---:|
| InternLM2.5-7B-chat (8-bit) | Without | kept | 0.3869 ± 0.0207 | 0.4050 | 0.3543 |
| InternLM2.5-7B-chat (8-bit) | With | kept | 0.6086 ± 0.0368 | 0.6507 | 0.6273 |
| InternLM2.5-7B-chat (8-bit) | Without | removed | 0.3707 ± 0.0205 | 0.4012 | 0.3356 |
| InternLM2.5-7B-chat (8-bit) | With | removed | 0.5846 ± 0.0292 | 0.6411 | 0.5985 |
| Gemma-2-27B-it (4-bit) | Without | kept | 0.4399 ± 0.0307 | 0.4319 | 0.4218 |
| Gemma-2-27B-it (4-bit) | With | kept | 0.5708 ± 0.0211 | 0.5374 | 0.5470 |

Removing the question lowers InternLM by only 0.02–0.03 test macro-F1, compared with 0.13 for Logistic Regression, so the question wording inflates the classical baselines much more than the LLM.

**Gemma prompt.** The reported Gemma results come from the v1 prompt, not the paper prompts InternLM uses, so the Gemma vs InternLM comparison mixes model and prompt.

### Uncertainty and Robustness

```bash
python src/export_test_predictions.py
python src/paired_bootstrap.py
python src/cluster_bootstrap_check.py
python src/country_grouped_cv.py
```

- **Bootstrap CIs.** `paired_bootstrap.py` resamples the 521 test items to give a 95% percentile CI for each model's macro-F1 and for the paired difference between two models on the same items.
- **Cluster check.** Test items come from 63 Background passages, so they are not independent. `cluster_bootstrap_check.py` resamples whole passages instead. CI bounds move by at most 0.0198, and only one conclusion changes: Gemma vs InternLM without the rule (+0.0675). Its item CI [0.0176, 0.1165] excludes 0, but its cluster CI [−0.0009, 0.1363] does not, so that difference is not significant.
- **Unseen countries.** `country_grouped_cv.py` reruns the classical CV with folds grouped by country, so validation countries never appear in training. Logistic Regression keeps its score (0.9031 without the rule, 0.9124 with, question kept), so its accuracy is not a country lookup.

### Main Results Table and Figures

```bash
python src/make_results.py
```

Reads whichever result CSVs exist and writes `results/main_table.csv` / `main_table.md` (CV mean ± SD, test accuracy, test macro-F1 with its 95% CI for every model and condition), `results/figures/fig1_test_macro_f1.png` and `results/figures/fig2_question_ablation.png`.

### Error Analysis

```bash
python src/error_analysis.py
```

Uses the committed test predictions only. Writes per-model error summaries, confusion matrices (CSV plus `confusion_baselines.svg` and `confusion_llms.svg`), per-country and per-subaxis errors, LLM error hotspots, and eight hand-picked error cases (`example_errors.csv`) to `results/error_analysis/`. The script checks that each case is still an error. Per-country scores rest on 3–26 items per country, so read them with care. See the [error-analysis write-up](results/error_analysis/README.md) for the findings.

### Reproducing Everything

Run from the repository root, in this order: `baselines.py`, `score_llms.py`, `export_test_predictions.py`, `paired_bootstrap.py`, `cluster_bootstrap_check.py`, `country_grouped_cv.py`, `make_results.py`, `error_analysis.py`. LLM inference itself runs in the notebooks (GPU), not in these scripts; the scripts score the saved predictions.

| Software | Version |
|---|---|
| Python | 3.14.3 |
| scikit-learn | 1.9.1 |
| pandas | 3.0.6 |
| Hugging Face Datasets | 5.0.1 |
| matplotlib | 3.11.2 |

All dependencies are pinned in `requirements.txt`.

The dataset is loaded from Hugging Face:
[akhilayerukola/NormAd](https://huggingface.co/datasets/akhilayerukola/NormAd), pinned to revision `bb1a2997b6ab74b1f901493b15a67c2250a366c9` in `src/data_pipeline.py`. Rerunning `src/baselines.py` with this revision reproduces every value in `results/baseline_results.csv` exactly.

## License and contributions

The project's original code is licensed under the [MIT License](LICENSE). The [NormAd dataset](https://huggingface.co/datasets/akhilayerukola/NormAd) retains its CC BY 4.0 license. See [CONTRIBUTIONS.md](CONTRIBUTIONS.md) for individual contributions.
