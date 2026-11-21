| Model | Rule-of-Thumb | Question | CV Macro-F1 (mean ± SD, 5 folds) | Test Accuracy | Test Macro-F1 [95% CI] |
|---|---|---|---:|---:|---:|
| Dummy (most frequent) | N/A | kept | — | 0.3647 | 0.1782 |
| Dummy (stratified, 10 seeds) | N/A | kept | — | 0.3326 | 0.3299 ± 0.0148 (seed SD) |
| TF-IDF + Logistic Regression | Without | kept | 0.9010 ± 0.0123 | 0.9194 | 0.9191 [0.8944, 0.9413] |
| TF-IDF + Logistic Regression | With | kept | 0.8991 ± 0.0177 | 0.9079 | 0.9078 [0.8822, 0.9313] |
| TF-IDF + Naive Bayes | Without | kept | 0.8131 ± 0.0247 | 0.8100 | 0.8083 [0.7730, 0.8414] |
| TF-IDF + Naive Bayes | With | kept | 0.7835 ± 0.0232 | 0.7908 | 0.7895 [0.7532, 0.8234] |
| InternLM2.5-7B-chat (8-bit) | Without | kept | 0.3869 ± 0.0207 | 0.4050 | 0.3543 [0.3180, 0.3905] |
| InternLM2.5-7B-chat (8-bit) | With | kept | 0.6086 ± 0.0368 | 0.6507 | 0.6273 [0.5870, 0.6686] |
| Gemma-2-27B-it (4-bit, v1 prompt) | Without | kept | 0.4399 ± 0.0307 | 0.4319 | 0.4218 [0.3776, 0.4645] |
| Gemma-2-27B-it (4-bit, v1 prompt) | With | kept | 0.5708 ± 0.0211 | 0.5374 | 0.5470 [0.5048, 0.5889] |
| TF-IDF + Logistic Regression | Without | removed | 0.7978 ± 0.0192 | 0.7908 | 0.7887 [0.7528, 0.8224] |
| TF-IDF + Logistic Regression | With | removed | 0.7974 ± 0.0155 | 0.8081 | 0.8086 [0.7747, 0.8413] |
| TF-IDF + Naive Bayes | Without | removed | 0.7626 ± 0.0240 | 0.7639 | 0.7613 [0.7242, 0.7971] |
| TF-IDF + Naive Bayes | With | removed | 0.7278 ± 0.0158 | 0.7409 | 0.7393 [0.7002, 0.7770] |
| InternLM2.5-7B-chat (8-bit) | Without | removed | 0.3707 ± 0.0205 | 0.4012 | 0.3356 [0.3030, 0.3667] |
| InternLM2.5-7B-chat (8-bit) | With | removed | 0.5846 ± 0.0292 | 0.6411 | 0.5985 [0.5576, 0.6394] |

Test set: 521 held-out items (grouped by Background). CV: the same 5 StratifiedGroupKFold folds for every model.
95% CIs: percentile bootstrap over test items, 10,000 resamples (src/paired_bootstrap.py).
Gemma uses a different prompt from InternLM (v1, see results/gemma-2-27b/full/gemma-2-27b_run-notes_full.txt) and was not run with the question removed.
