# Milestone 5: Where the models went wrong

All comparisons below use the same 521 held-out NormAd stories: 190 labeled **yes**, 174 **no**, and 157 **neither**. A mistake is any prediction that differs from the dataset label. A reply with no usable label counts as a mistake too.

## What changed when the rule was supplied?

With the original question kept, the two LLMs often answered **neither** when the gold label was **no**. Giving the model the story's rule of thumb reduced this mistake and reduced total errors:

| Model | Errors without rule | Errors with rule | Gold no → neither, without rule | Gold no → neither, with rule |
| --- | ---: | ---: | ---: | ---: |
| InternLM 2.5 7B | 310/521 | 182/521 | 157/174 | 115/174 |
| Gemma 2 27B | 296/521 | 241/521 | 122/174 | 85/174 |

Gemma also answered **neither** for 120/190 gold **yes** stories without the rule, versus 74/190 with it. Its unparseable replies (3 without the rule, 5 with it) are a separate column in the matrices. For comparison, logistic regression without the rule made 42/521 errors with the original question and 109/521 after the question was removed. Those are different input conditions; the 42-error result should not be read as performance without the question wording.

[LLM confusion matrices](confusion_llms.svg) · [Classical and dummy matrices](confusion_baselines.svg) · [Exact counts for every matrix](confusion_matrices.csv) · [Scores and error totals](model_summary.csv)

## Which countries and norms had the highest error rates?

Each cell shows **errors / test stories**. To avoid picking a group just because the model missed its only story, the rate ranking requires at least 10 stories. This leaves 23 of 43 countries and 5 of 13 norm categories (the dataset's “subaxes”). All groups, including smaller ones, remain in the detailed tables.

| Model | Rule | Question | Highest country error rate | Highest subaxis error rate |
| --- | --- | --- | --- | --- |
| InternLM | Without | Kept | Sudan: 13/16 (81%) | Offering and complimenting items: 15/18 (83%) |
| InternLM | With | Kept | Sudan: 10/16 (63%) | Offering and complimenting items: 9/18 (50%) |
| Gemma | Without | Kept | Brazil: 19/20 (95%) | Offering and complimenting items: 12/18 (67%) |
| Gemma | With | Kept | Afghanistan: 7/10; Brazil: 14/20; Philippines: 14/20 (all 70%) | Gift giving: 59/113 (52%) |
| InternLM | Without | Removed | Palestinian Territories: 12/14 (86%) | Offering and complimenting items: 14/18 (78%) |
| InternLM | With | Removed | North Macedonia: 6/10; Philippines: 12/20 (both 60%) | Visiting: 45/104 (43%) |

**Error counts answer a different question.** With the question kept and no rule, basic etiquette accounted for the most mistakes for *both* LLMs (76/129 each), even though offering and complimenting items had the higher rate. With the rule, gift giving accounted for the most mistakes (48/113 for InternLM; 59/113 for Gemma). Brazil's 20 stories come from just three backgrounds and Sudan's 16 from two. These rankings describe this test split; they are weak evidence about either model's performance for a country in general.

[Every model by country](by_country.csv) · [Every model by subaxis](by_subaxis.csv) · [Largest counts and highest eligible rates for all LLM settings, including ties](llm_hotspots.csv)

## What do individual mistakes look like?

The [eight checked examples](example_errors.csv) include the story ID, configuration, gold and predicted labels, source prediction file, and a possible reason for each mistake. IDs can be looked up in [the cleaned dataset](../../data/normad_clean.csv).

- **Missing context (ID 1403, Fiji):** A guest starts eating when the food arrives. InternLM answered **neither** without the rule; with the instruction to wait for the group and the blessing, it correctly answered **no**.
- **Incomplete response (ID 889, Cambodia):** The story describes returning a scarf with both hands. Gemma's entire saved reply was “Mike's action is”. It gave no label, so the intended answer is unknown.
- **Conflicting annotation (ID 1609, New Zealand):** Logistic regression predicted **no** and the recorded gold label is **yes**, but the dataset's own explanation concludes that the action was unacceptable. We kept the published label for scoring and flagged the disagreement.

These cases illustrate different error types; their frequency in this small selection is not a prevalence estimate. The explanations are interpretations of the text and outputs, not claims about a model's internal reasoning.

## Reproduce and read the tables

From the repository root, run `python src/error_analysis.py`. It reads committed predictions and metadata; no model download or inference is needed. It recreates the project's 20% group split (seed 42, grouped by background), checks the gold labels, and joins on story ID. There are 63 distinct test backgrounds. The six full LLM exports are filtered to the identical test IDs used by the classical models. Gemma has no saved question-removed run. The script treats `neutral` as `neither`, as in the project's scorer.

There are **14 configurations with prediction CSVs plus two dummy baselines**. The project saved only aggregate scores for the dummies, so their predictions are reproduced with the original strategies, seeds and test order. The stratified dummy's counts pool ten seeds (5,210 predictions on the same 521 stories); its macro-F1 is the mean across seeds. For all other configurations each story contributes one prediction. In the subgroup files, `n_stories`, `n_backgrounds`, and `n_predictions` give the denominators and context; `error_rate` is `errors / n_predictions`. Subaxis names are kept as supplied, including the separate `gifts` and `gift_giving` categories.
