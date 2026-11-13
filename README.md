# NLP CS60075 Assignment II - Project 5
## From Static to Contextual: Distributional Semantics and What's Inside BERT

**Student**: Pilla Srujan Narayana (Roll No: `24CS10022`) , Verma Pratik (Roll No: `24CS10138`)  
**Group**: Group 14  
**Assigned Dataset**: Universal Dependencies English GUM (`universal_dependencies`, config `en_gum`)  
**Shared Evaluation Benchmarks**: WordSim-353, Google Analogy Set (`questions-words.txt`)

---

## 1. Project Overview

This project explores the fundamental paradigm shift from classical static distributional semantics to deep contextual representations, and investigates whether syntactic structure is intrinsically encoded within pretrained transformers:
- **Task 1: Baseline (PPMI + SVD Static Word Vectors)**: Constructs a classical distributional co-occurrence space (symmetric window size 5) from the training split of UD English GUM, applies Positive Pointwise Mutual Information (PPMI), and reduces to 100-dimensional dense embeddings via Truncated SVD. Evaluated on WordSim-353 (Spearman rank correlation), Google Analogy Set (3CosAdd vector offset), and a 20-sentence Minimum Spanning Tree (MST) structural check.
- **Task 2: Paper Reproduction (Structural Probe on BERT)**: Implements the structural distance and depth probes of *Hewitt & Manning (NAACL 2019)* across all 12 layers of frozen `bert-base-uncased` across ranks $k \in \{32, 64, 128\}$. Aligns word tokens with WordPiece subwords, trains against gold dependency distances and depths from GUM annotations, evaluates Undirected Unlabeled Attachment Score (UUAS) via MST decoding and depth Spearman correlation, identifying the peak syntax-encoding layer $\ell^*$.
- **Task 3: Combine & Improve (BERT-Derived Static Vectors)**: Pools contextual representations from the identified optimal layer $\ell^*$ across all occurrences in the GUM training split into a static word embedding table, testing whether syntax-rich contextual representations survive averaging and surpass classical static vectors on semantic and structural benchmarks.

---

## 2. Directory Structure

```
project5/
├── README.md
├── requirements.txt
├── config.py
├── main.py
│
├── task1/
│   ├── dataset.py
│   ├── cooccurrence.py
│   ├── ppmi.py
│   ├── svd_embeddings.py
│   ├── evaluation.py
│   ├── structural_check.py
│   └── main.py
│
├── task2/
│   ├── dataset.py
│   ├── alignment.py
│   ├── gold_structures.py
│   ├── bert_extractor.py
│   ├── distance_probe.py
│   ├── depth_probe.py
│   ├── tree_decoder.py
│   ├── evaluation.py
│   └── main.py
│
├── task3/
│   ├── embedding_extractor.py
│   ├── static_vectors.py
│   ├── evaluation.py
│   ├── structural_check.py
│   └── main.py
│
├── common/
│   ├── data_utils.py
│   ├── metrics.py
│   ├── visualization.py
│   └── reproducibility.py
│
├── scripts/
│   ├── download_data.py
│   └── run_all.py
│
├── tests/
│   ├── test_preprocessing.py
│   ├── test_embeddings.py
│   ├── test_probes.py
│   └── test_metrics.py
│
├── results/
│   ├── task1/
│   ├── task2/
│   └── task3/
│
├── checkpoints/
│   ├── task1/
│   ├── task2/
│   └── task3/
│
└── report/
    ├── report.md
    ├── report.pdf
    └── figures/
```

---

## 3. Environment Setup & Installation

### Requirements
- Python >= 3.10
- GPU (Optional but recommended): NVIDIA GPU with CUDA support (e.g. RTX 3050 / T4). The code automatically falls back to CPU if CUDA is unavailable.

### Virtual Environment Setup
```bash
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

---

## 4. Downloading Data & Pretrained Models

To download the UD English GUM treebank, WordSim-353, Google Analogy questions, and the `bert-base-uncased` checkpoint:
```bash
python scripts/download_data.py
```
Or use the CLI flag:
```bash
python main.py --download
```

---

## 5. Execution Guide

### Option A: Run End-to-End (All Tasks Sequentially)
```bash
python main.py --task all
```
or:
```bash
python scripts/run_all.py
```

### Option B: Run Tasks Independently

#### Run Task 1 (PPMI + SVD Static Vectors):
```bash
python main.py --task 1
# or
python -m task1.main
```
Outputs generated in `results/task1/` and checkpoints in `checkpoints/task1/`.

#### Run Task 2 (Structural Probe on BERT):
```bash
python main.py --task 2
# or
python -m task2.main
```
Outputs generated in `results/task2/`, checkpoints in `checkpoints/task2/`, and plots in `report/figures/`.

#### Run Task 3 (BERT-Derived Static Vectors & Comparison):
```bash
python main.py --task 3
# or
python -m task3.main
```
Outputs generated in `results/task3/` and comparison plot in `report/figures/`.

---

## 6. Running Unit Tests

Run the test suite to verify math operations, dimension invariants, probes, and metrics:
```bash
python -m unittest discover tests
```

---

## 7. Outputs & Artifacts

- **Task 1 Results**:
  - `results/task1/task1_results.json`: Full metrics for WordSim-353, Google Analogy, and 20-sentence MST check.
  - `checkpoints/task1/vocab.json`, `svd_embeddings.npy`, `cooccurrence_matrix.npz`, `ppmi_matrix.npz`.
- **Task 2 Results**:
  - `results/task2/task2_results.json`: Layer-wise UUAS and depth Spearman correlation across all ranks and layers.
  - `checkpoints/task2/best_layer_info.json`: Identified optimal layer $\ell^*$.
  - `report/figures/task2_uuas_vs_layer.png`: Plot of UUAS vs. BERT Layer.
  - `report/figures/task2_depth_corr_vs_layer.png`: Plot of Depth Correlation vs. BERT Layer.
- **Task 3 Results**:
  - `results/task3/task3_results.json`: Evaluated metrics on BERT-derived static vectors.
  - `results/task3/comparison_table.csv`: Side-by-side comparison between Task 1 and Task 3.
  - `report/figures/task1_vs_task3_comparison.png`: Comparison bar chart.
- **Academic Report**:
  - `report/report.md` & `report/report.pdf`: 8–12 page academic report following CS60075 guidelines.

---

## 8. Reproducibility & Random Seeds

All experiments use fixed random seeds (`seed = 42`) for:
- Python standard library `random`
- NumPy `np.random`
- PyTorch CPU and CUDA generators (`torch.manual_seed(42)`)
- Truncated SVD randomized solver
- The 20-sentence test sample is saved to `data/structural_check_test_indices.json` and reused identically between Task 1 and Task 3.
