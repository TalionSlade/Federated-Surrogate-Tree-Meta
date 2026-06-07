"""Reproduce DNDT across the bundled datasets in one command.

    python run_all.py

Calls train.py for iris, wine, and breast_cancer (the last with a top-6
feature subset to keep the leaf count tractable) and prints a summary table.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

RUNS = [
    ["--dataset", "iris", "--n_cut", "1", "--epochs", "1000"],
    ["--dataset", "wine", "--n_cut", "1", "--epochs", "2000"],
    ["--dataset", "breast_cancer", "--n_cut", "1", "--epochs", "2000", "--feature_topk", "6"],
]


def main() -> None:
    for args in RUNS:
        cmd = [sys.executable, str(HERE / "train.py"), *args]
        print("\n" + "=" * 70)
        print(" ".join(cmd))
        print("=" * 70)
        subprocess.run(cmd, check=True, cwd=HERE)


if __name__ == "__main__":
    main()
