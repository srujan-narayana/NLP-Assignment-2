import os
from pathlib import Path

# Enable offline mode for cached datasets and models by default
if "HF_HUB_OFFLINE" not in os.environ:
    os.environ["HF_HUB_OFFLINE"] = "1"
if "TRANSFORMERS_OFFLINE" not in os.environ:
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

# Set writable matplotlib cache directory
mpl_cache = Path(__file__).resolve().parent / "data" / "matplotlib"
mpl_cache.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(mpl_cache)

# Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
CHECKPOINTS_DIR = BASE_DIR / "checkpoints"
RESULTS_DIR = BASE_DIR / "results"
REPORT_DIR = BASE_DIR / "report"
FIGURES_DIR = REPORT_DIR / "figures"

# Create directories if not exist
for d in [DATA_DIR, CHECKPOINTS_DIR, RESULTS_DIR, REPORT_DIR, FIGURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

(RESULTS_DIR / "task1").mkdir(parents=True, exist_ok=True)
(RESULTS_DIR / "task2").mkdir(parents=True, exist_ok=True)
(RESULTS_DIR / "task3").mkdir(parents=True, exist_ok=True)

# General Settings
RANDOM_SEED = 42

# Dataset Configuration
DATASET_NAME = "universal-dependencies/universal_dependencies"
DATASET_CONFIG = "en_gum"

# Task 1 Hyperparameters
WINDOW_SIZE = 5
EMBEDDING_DIM = 100
MIN_WORD_FREQ = 1
STRUCTURAL_SAMPLE_SIZE = 20
STRUCTURAL_SAMPLE_SEED = 42

# Task 2 Hyperparameters
BERT_MODEL_NAME = "bert-base-uncased"
PROBE_LAYERS = list(range(1, 13))  # Layers 1 to 12
PROBE_RANKS = [32, 64, 128]
PROBE_LR = 0.001
PROBE_BATCH_SIZE = 20
PROBE_EPOCHS = 30
MAX_SEQ_LEN = 128

# WordSim-353 and Analogy Configuration
WORDSIM353_DATASET = "enus5three/WordSim353"
GOOGLE_ANALOGY_URL = "https://raw.githubusercontent.com/tmikolov/word2vec/master/questions-words.txt"
GOOGLE_ANALOGY_PATH = DATA_DIR / "questions-words.txt"
