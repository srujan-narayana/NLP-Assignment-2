import tempfile
from pathlib import Path


_TEMP_ROOT = Path(__file__).resolve().parents[2] / ".test_tmp"
_TEMP_ROOT.mkdir(parents=True, exist_ok=True)


def temporary_directory():
    return tempfile.TemporaryDirectory(dir=_TEMP_ROOT)
