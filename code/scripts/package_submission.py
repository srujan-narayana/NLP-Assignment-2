import zipfile
from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ZIP_OUTPUT_PATH = PROJECT_ROOT / "project5_submission.zip"

def create_submission_zip():
    print(f"Creating submission package at {ZIP_OUTPUT_PATH}...")
    
    # Files and directories to include
    include_map = [
        # (source_path, archive_dest_path)
        (PROJECT_ROOT / "task1", "code/task1"),
        (PROJECT_ROOT / "task2", "code/task2"),
        (PROJECT_ROOT / "task3", "code/task3"),
        (PROJECT_ROOT / "common", "code/common"),
        (PROJECT_ROOT / "scripts", "code/scripts"),
        (PROJECT_ROOT / "tests", "code/tests"),
        (PROJECT_ROOT / "config.py", "code/config.py"),
        (PROJECT_ROOT / "main.py", "code/main.py"),
        (PROJECT_ROOT / "README.md", "README.md"),
        (PROJECT_ROOT / "requirements.txt", "requirements.txt"),
        (PROJECT_ROOT / "report" / "report.pdf", "report.pdf"),
        (PROJECT_ROOT / "report" / "report.md", "report/report.md"),
        (PROJECT_ROOT / "results" / "task1" / "task1_results.json", "results/task1/task1_results.json"),
        (PROJECT_ROOT / "results" / "task2" / "task2_results.json", "results/task2/task2_results.json"),
        (PROJECT_ROOT / "results" / "task3" / "task3_results.json", "results/task3/task3_results.json"),
        (PROJECT_ROOT / "results" / "task3" / "comparison_table.csv", "results/task3/comparison_table.csv"),
    ]

    # Include figures in report/figures
    figures_dir = PROJECT_ROOT / "report" / "figures"
    if figures_dir.exists():
        for fig in figures_dir.glob("*.png"):
            include_map.append((fig, f"report/figures/{fig.name}"))

    with zipfile.ZipFile(ZIP_OUTPUT_PATH, "w", zipfile.ZIP_DEFLATED) as zf:
        for src, arcname in include_map:
            if src.is_dir():
                for root, _, files in os.walk(src):
                    for file in files:
                        if file.endswith((".pyc", ".pt", ".npy", ".npz", ".pkl")) or "__pycache__" in root:
                            continue
                        full_path = Path(root) / file
                        rel_path = full_path.relative_to(src)
                        zf.write(full_path, f"{arcname}/{rel_path}")
            elif src.is_file():
                zf.write(src, arcname)

    size_mb = ZIP_OUTPUT_PATH.stat().st_size / (1024 * 1024)
    print(f"Submission archive successfully created: {ZIP_OUTPUT_PATH} ({size_mb:.2f} MB)")

    # Print archive structure summary
    with zipfile.ZipFile(ZIP_OUTPUT_PATH, "r") as zf:
        namelist = zf.namelist()
        print(f"Total files in archive: {len(namelist)}")
        for name in sorted(namelist)[:25]:
            print("  -", name)
        if len(namelist) > 25:
            print(f"  ... and {len(namelist) - 25} more files.")

if __name__ == "__main__":
    create_submission_zip()
