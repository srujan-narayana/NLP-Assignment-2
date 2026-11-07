import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import logging
import time
from scripts.download_data import download_all
from task1.main import run_task1
from task2.main import run_task2
from task3.main import run_task3

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("RunAll")

def main():
    start_time = time.time()
    logger.info("=================================================================")
    logger.info("   NLP CS60075 Assignment II - Project 5 Complete Execution    ")
    logger.info("=================================================================")

    # Step 0: Ensure all data & models
    download_all()

    # Step 1: Run Task 1
    t1_start = time.time()
    logger.info(">>> Launching Task 1: PPMI + SVD Static Vectors <<<")
    run_task1()
    logger.info(f">>> Task 1 completed in {time.time() - t1_start:.2f}s <<<")

    # Step 2: Run Task 2
    t2_start = time.time()
    logger.info(">>> Launching Task 2: Structural Probe on BERT <<<")
    best_layer, _ = run_task2()
    logger.info(f">>> Task 2 completed in {time.time() - t2_start:.2f}s <<<")

    # Step 3: Run Task 3
    t3_start = time.time()
    logger.info(">>> Launching Task 3: BERT-Derived Static Vectors <<<")
    run_task3(selected_layer=best_layer)
    logger.info(f">>> Task 3 completed in {time.time() - t3_start:.2f}s <<<")

    total_time = time.time() - start_time
    logger.info(f"=================================================================")
    logger.info(f"   Project 5 Execution Fully Finished in {total_time:.2f}s!    ")
    logger.info(f"=================================================================")

if __name__ == "__main__":
    main()
