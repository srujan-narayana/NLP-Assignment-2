# Project 5 Results

## Provenance and interpretation

The raw files in reference_results/results/ were copied unchanged from the supplied results.zip. They are outputs from an earlier implementation, not a fresh run of the unified code in this repository. The full UD English GUM data and bert-base-uncased checkpoint were not available in the environment used to prepare this repository.

The earlier Task 2 result file contains test metrics but no validation metrics. The earlier implementation selected its layer and rank using test scores, so its selected layer 8/rank 64 must not be treated as validation-selected by the unified implementation. The unified code selects on validation UUAS and reports test scores separately.

The earlier Task 1 and Task 3 analogy scores were produced before the unified code's normalized-query 3CosAdd correction. Re-run the unified pipeline to get comparable corrected scores. The prior project report also gave a different Task 2 depth correlation for rank 64/layer 8 (0.7171) than the archived Task 2 JSON (0.7600). The JSON is preserved as supplied; neither value has been changed or reconciled here.

## Task 1: PPMI + SVD static vectors

| Metric | Supplied result |
|---|---:|
| Vocabulary size | 16,763 |
| Embedding dimension | 100 |
| WordSim-353 Spearman rho | 0.1559 |
| WordSim-353 coverage | 268 / 353 (85 OOV pairs) |
| Google analogy accuracy | 0.0090 (56 / 6,231 evaluated) |
| Analogy questions dropped as OOV | 13,313 / 19,544 |
| Structural MST attachment agreement | 0.1543 (48 / 311 edges) |
| Structural macro agreement | 0.2837 |
| Structural sample size | 20 sentences |

## Task 2: BERT structural probes

The archived JSON records layer 8, rank 64 as its selected configuration. At that configuration, the archived test UUAS is 0.5799 and depth Spearman is 0.7600. The full layer-by-rank test results are below. Each cell is test UUAS / test depth Spearman.

| Layer | Rank 32 | Rank 64 | Rank 128 |
|---:|---:|---:|---:|
| 1 | 0.4393 / 0.6050 | 0.4239 / 0.5861 | 0.4086 / 0.5698 |
| 2 | 0.4638 / 0.5942 | 0.4430 / 0.5987 | 0.4310 / 0.5812 |
| 3 | 0.4918 / 0.6240 | 0.4868 / 0.6244 | 0.4566 / 0.6247 |
| 4 | 0.5317 / 0.6805 | 0.5312 / 0.6373 | 0.5098 / 0.6499 |
| 5 | 0.5571 / 0.6878 | 0.5383 / 0.6689 | 0.5203 / 0.6854 |
| 6 | 0.5629 / 0.7155 | 0.5472 / 0.7114 | 0.5515 / 0.7157 |
| 7 | 0.5653 / 0.7352 | 0.5584 / 0.7111 | 0.5335 / 0.7078 |
| 8 | 0.5667 / 0.7626 | 0.5799 / 0.7600 | 0.5520 / 0.6991 |
| 9 | 0.5514 / 0.7519 | 0.5300 / 0.7267 | 0.5374 / 0.7288 |
| 10 | 0.5050 / 0.6921 | 0.4986 / 0.7006 | 0.4890 / 0.6872 |
| 11 | 0.4695 / 0.6521 | 0.4649 / 0.6560 | 0.4557 / 0.6572 |
| 12 | 0.4752 / 0.6727 | 0.4694 / 0.6772 | 0.4632 / 0.6555 |

## Task 3: BERT-derived static vectors

| Metric | Supplied result |
|---|---:|
| Selected BERT layer | 8 |
| Vocabulary size | 6,204 |
| WordSim-353 Spearman rho | 0.5155 |
| WordSim-353 coverage | 113 / 353 (240 OOV pairs) |
| Google analogy accuracy | 0.6533 (912 / 1,396 evaluated) |
| Analogy questions dropped as OOV | 18,148 / 19,544 |
| Structural MST attachment agreement | 0.0772 (24 / 311 edges) |
| Structural macro agreement | 0.2043 |
| Structural sample size | 20 sentences |

## Supplied Task 1 vs Task 3 comparison

| Metric | Task 1 | Task 3 | Delta |
|---|---:|---:|---:|
| WordSim-353 Spearman rho | 0.1559 | 0.5155 | +0.3596 |
| Google analogy accuracy | 0.0090 | 0.6533 | +0.6443 |
| Structural MST agreement | 0.1543 | 0.0772 | -0.0772 |

Raw JSON and CSV files are in reference_results/results/. Run the unified pipeline to replace these reference values with results produced by the current implementation.
