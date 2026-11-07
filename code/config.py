from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
FIGURES_DIR = RESULTS_DIR / "figures"
CACHE_DIR = DATA_DIR / "cache" / "bert"

DATASET_REPOSITORIES = (
    "universal_dependencies",
    "universal-dependencies/universal_dependencies",
)
DATASET_CONFIG = "en_gum"
WORDSIM_DATASET = "enus5three/WordSim353"
GOOGLE_ANALOGY_URL = (
    "https://raw.githubusercontent.com/tmikolov/word2vec/master/questions-words.txt"
)
BERT_MODEL_NAME = "bert-base-uncased"

RANDOM_SEED = 42
WINDOW_SIZE = 5
EMBEDDING_DIM = 100
MIN_WORD_FREQ = 1
STRUCTURAL_SAMPLE_SIZE = 20
STRUCTURAL_SAMPLE_SEED = 42
PROBE_LAYERS = tuple(range(1, 13))
PROBE_RANKS = (32, 64, 128)
PROBE_LR = 1e-3
PROBE_BATCH_SIZE = 20
PROBE_EPOCHS = 30
MAX_BERT_TOKENS = 512
