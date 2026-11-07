import sys
import os
from pathlib import Path

# Disable offline mode for downloading
os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "0"

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import logging
from common.data_utils import load_ud_gum_dataset, ensure_google_analogy_dataset, load_wordsim353_dataset
from transformers import BertTokenizer, BertModel
from config import BERT_MODEL_NAME

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("DownloadData")

def download_all():
    logger.info("=== Downloading and Preparing All Datasets & Checkpoints ===")
    
    # 1. UD English GUM
    logger.info("1/4: Downloading UD English GUM...")
    load_ud_gum_dataset()

    # 2. Google Analogy
    logger.info("2/4: Downloading Google Analogy questions-words.txt...")
    ensure_google_analogy_dataset()

    # 3. WordSim-353
    logger.info("3/4: Downloading WordSim-353 benchmark...")
    load_wordsim353_dataset()

    # 4. Pretrained BERT
    logger.info(f"4/4: Downloading pretrained tokenizer and model: {BERT_MODEL_NAME}...")
    BertTokenizer.from_pretrained(BERT_MODEL_NAME)
    BertModel.from_pretrained(BERT_MODEL_NAME)

    logger.info("=== All resources successfully downloaded and cached! ===")

if __name__ == "__main__":
    download_all()
