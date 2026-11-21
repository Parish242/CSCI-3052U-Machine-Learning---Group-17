# Contributions

Each group member can add their own section here. Entries below are based on the repository's commit history.

## Sam Nguyen

- **Proposal and setup (Milestone 1 and 2):** Updated the project proposal. Ran InternLM2.5-7B-chat on the 50-item trial subset and then the full dataset, and uploaded the outputs to `results/internlm2.5-7b/` (Lee Language Lab project task).
- **Milestone 3:** Wrote the [data pipeline](src/data_pipeline.py) (pinned NormAd revision, Background-grouped train/test split, shared CV folds) and the first versions of the [baselines](src/baselines.py), including the grid search. Reorganized the repository layout, added Parish's full Gemma outputs to `results/gemma-2-27b/full/`, and checked that the baseline results reproduce exactly.
- **Milestone 4:** Added the question ablation to the baselines (`results/baseline_results_no_question.csv`). Wrote [score_llms.py](src/score_llms.py) to score LLM predictions on the same test set and CV folds as the baselines. Added the [InternLM inference notebook](notebooks/normad_internlm2_5_7b_inference.ipynb), reran InternLM with the question removed, and created [AI_USE.md](AI_USE.md).
- **Milestone 5:** Saved every grid-search configuration to `results/tuning/`. Added the [cluster bootstrap check](src/cluster_bootstrap_check.py) and question-removed runs to the country-grouped CV. Added the missing bootstrap pairs and the Gemma vs InternLM comparison to [paired_bootstrap.py](src/paired_bootstrap.py). Added the [Gemma v1-prompt notebook](notebooks/Gemma-27B_v1-prompt.ipynb), aligned Gemma to the same 521 test IDs, and documented the Gemma prompt mismatch and 5-token reply cap. Wrote [make_results.py](src/make_results.py) and [plot_style.py](src/plot_style.py) for the main results table and Figures 1–2, and updated the README.

## Paul Joseph

- **Proposal and setup (Milestone 1 and 2):** Wrote the [exploratory data analysis notebook](notebooks/normad_eda.ipynb) and the first `requirements.txt` and `.gitignore`.
- **Milestone 3:** Completed the baseline evaluation in [baselines.py](src/baselines.py), exported `results/baseline_results.csv`, and documented the classical baselines and their results in the README.
- **Milestone 5:** Added [export_test_predictions.py](src/export_test_predictions.py) (per-item classical test predictions), [paired_bootstrap.py](src/paired_bootstrap.py) (95% bootstrap CIs and paired model differences), and [country_grouped_cv.py](src/country_grouped_cv.py) (robustness to unseen countries). Extended the bootstrap to cover the question ablation.

## Parish Schitka

- **Proposal and setup (Milestone 1 and 2):** Created the group repository and uploaded the 50-item Gemma-2-27B trial run.
- Ran Gemma-2-27B-it (4-bit, v1 prompt) on the full dataset with and without the rule-of-thumb. These are the Gemma predictions, classification reports and run notes in `results/gemma-2-27b/full/`. (Lee Language Lab project task).
- **Milestone 5:** Built the [Gemma-2-27B notebook](notebooks/Gemma-27B.ipynb) (Kaggle, 2 GPUs) with the paper prompts and question stripping, plus a Gemma EDA notebook that was later removed. Revised the Gemma full-run notes in `results/gemma-2-27b/full/`.

## Caelan Schweizer

- **Milestone 5:** Added the [error-analysis script](src/error_analysis.py) and [results](results/error_analysis/README.md) for the shared test split. The analysis covers confusion matrices, errors by country and norm category, and comparisons with the dummy baselines.
- Documented eight prediction errors by story ID, including an incomplete model reply and an apparent disagreement between a gold label and its explanation.
- Added the MIT license for the project's original code.
