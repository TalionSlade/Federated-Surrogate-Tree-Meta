"""Reproduce NODE across the bundled datasets in one command.

    python run_all.py

Calls train.py for iris, wine, and breast_cancer. Unlike DNDT, NODE handles
high-dimensional inputs natively (sparse entmax feature selection), so no
feature subsetting is required.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

RUNS = [
    ["--dataset", "iris", "--epochs", "80"],
    ["--dataset", "wine", "--epochs", "100"],
    ["--dataset", "breast_cancer", "--epochs", "100"],
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
