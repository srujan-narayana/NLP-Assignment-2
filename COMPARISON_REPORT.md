# Implementation Comparison and Design Report

## Requirements used

Project 5 requires three connected tasks on the group's assigned UD English GUM treebank: (1) symmetric-window PPMI vectors reduced to 100 dimensions, with WordSim-353, Google analogy, and a 20-sentence dependency MST check; (2) frozen bert-base-uncased distance and depth probes over layers 1–12 and ranks 32, 64, and 128, trained for 30 epochs with Adam at 1e-3 and batches of 20 sentences; and (3) static BERT vectors pooled from the best syntax layer, evaluated with the same semantic benchmarks and the same 20 test sentences.

## Comparison

| Area | Version A: NLP_BERT_syntax_trees-main | Version B: project5_submission |
|---|---|---|
| Organization | Small modules for each task; little integration or shared configuration. | Clear task/common/scripts/tests split, configurable outputs, cached features, and a single runner. |
| Task 1 | Implements vocabulary, co-occurrence, PPMI, SVD, and benchmark functions. Its run uses a window of 2, and it has no structural check. | Implements the specified window of 5, sparse artifacts, benchmark reports, OOV counts, and the structural check. |
| Task 2 | Has the correct linear distance/depth probe formulas and a basic MST evaluator. | Adds word-to-subword mean pooling, train/validation/test handling, cache files, layer/rank results, plots, and checkpoint output. |
| Task 3 | Pools BERT occurrences and evaluates the two semantic benchmarks, but hardcodes layer 8 and omits the structural comparison. | Uses the Task 2 layer and includes the same-sample structural check and side-by-side results. |
| Tests and docs | No test suite; short README; loose dependency ranges. | Eleven unit tests and extensive documentation/report artifacts. The original tests passed in this environment. |
| Main weaknesses | Incomplete workflow, incorrect window default, arbitrary sentence-length filtering in the probe reader, and missing Task 3 requirements. | Test-set layer selection, silent BERT truncation misalignment, high memory use during pooling/cache loading, unnormalized analogy offsets, and inconsistent saved/report depth figures. |

## What the original checks established

- Version B's original unittest suite ran successfully: 11 tests passed. Those tests checked basic vector math, PPMI non-negativity, probe output shapes, and simple metrics; they did not detect the integration and data-alignment issues listed above.
- Version A has no automated tests. Its smoke script attempts to load bert-base-uncased and is not a data-free unit test.
- Version B's archived report gives rank-64, layer-8 depth correlation as 0.7171, while its saved task2_results.json gives 0.7600 for that same rank and layer. The report and result artifacts therefore do not establish one consistent, reproducible result. The final project does not copy those metric claims.

## Retained

- From Version A: the direct linear structural-probe geometry, simple PPMI/SVD decomposition, and explicit layer indexing for BERT hidden states.
- From Version B: the separate task modules, common metric/data helpers, result/checkpoint artifacts, validation split usage, BERT wordpiece mean pooling, plotting, shared structural sample, CLI workflow, and test coverage.

## Redesigned or improved

1. **Task 1 co-occurrence:** a sparse vectorized window builder uses the required symmetric window of 5 and preserves original word positions around filtered vocabulary items. PPMI handles empty matrices, and SVD always returns the requested output width, including small-vocabulary cases.
2. **Benchmark evaluation:** WordSim-353 reports coverage and OOV counts; Google analogies use normalized 3CosAdd inputs, exclude the three query words, and score eligible questions in batches.
3. **Dependency structures:** gold trees are checked for one root, legal heads, connectivity, and self-links. A deterministic Kruskal MST avoids unstable tie behavior and external graph objects.
4. **Task 2 selection:** every configured layer/rank is reported, but validation UUAS selects the layer. Validation depth correlation breaks ties. The test split is reserved for the requested reported metrics.
5. **BERT alignment and cache:** long inputs are split at word boundaries and every original word is represented. Cache keys include sentence content and model revision; each BERT layer is stored on disk and loaded one layer at a time instead of retaining all splits and layers in RAM.
6. **Task 3 pooling:** token vectors are accumulated as per-word sums/counts rather than storing a list of every occurrence. Task 1 and Task 3 use one saved structural test sample.
7. **Execution and docs:** configuration no longer changes Hugging Face offline settings at import time. The CLI supports local CoNLL-U input, explicit offline mode, selected devices, and smaller probe sweeps for smoke checks.

## Performance and resource use

- Co-occurrence construction processes each context offset with NumPy arrays and lets SciPy combine duplicate sparse entries, reducing Python-level pair bookkeeping.
- Analogy candidates are scored in batches with matrix multiplication.
- Task 2 caches word-level features in per-layer memory-mapped files. Training reads one layer at a time, bounding feature RAM by the three split sizes for one layer.
- Task 3 pooling uses O(vocabulary × dimension) accumulator memory instead of O(token occurrences × dimension).
- The full Task 2 experiment is still substantial: it trains two probes for every layer/rank combination for 30 epochs. A GPU is recommended.

## Verification and limitations

The final tests and local smoke run are recorded in the completion message. The local smoke run uses a tiny synthetic CoNLL-U fixture and does not substitute for the full assigned GUM experiment. Running the complete GUM/BERT workflow requires the dataset, shared benchmark files, and bert-base-uncased checkpoint. The sandbox could not resolve package-host DNS during the first dependency-helper attempt; approved network access later enabled the temporary PDF reader, but the GUM corpus and BERT checkpoint were not downloaded or trained here.

The repository keeps the supplied numerical outputs separately under reference_results/ and labels them as archived reference results. They are not copied into the default results/ output path, so the unified application cannot mistake them for fresh outputs. Run the complete workflow on the full GUM splits to generate current metrics. A small configured layer/rank/epoch subset is useful for smoke checks only and is not a reproduction of the assignment experiment.
