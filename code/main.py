import argparse
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from scripts.download_data import download_all
from task1.main import run_task1
from task2.main import run_task2
from task3.main import run_task3

def main():
    parser = argparse.ArgumentParser(description="NLP CS60075 Assignment II - Project 5")
    parser.add_argument("--task", type=str, default="all", choices=["1", "2", "3", "all"],
                        help="Task to execute (1, 2, 3, or all)")
    parser.add_argument("--download", action="store_true", help="Download all datasets and models")

    args = parser.parse_args()

    if args.download:
        download_all()

    if args.task in ["1", "all"]:
        run_task1()

    best_layer = None
    if args.task in ["2", "all"]:
        best_layer, _ = run_task2()

    if args.task in ["3", "all"]:
        run_task3(selected_layer=best_layer)

if __name__ == "__main__":
    main()
