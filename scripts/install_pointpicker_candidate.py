"""Temporary development/CI bootstrap until mpl-pointpicker reaches PyPI.

This URL is deliberately outside project.dependencies. Remove this bootstrap
after the publication prerequisite recorded in SCCircuits issue #83 is met.
"""

from pathlib import Path
import subprocess
import sys


CANDIDATE_REVISION = "9142ffc09123a9e9fe6e191180029bc226456d4f"
CANDIDATE_URL = (
    "https://github.com/joanjcaceres/mpl-pointpicker/archive/"
    f"{CANDIDATE_REVISION}.tar.gz"
)


if __name__ == "__main__":
    cache = Path(__file__).resolve().parents[1] / ".pixi" / "pip-cache"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--cache-dir",
            str(cache),
            "--no-deps",
            f"mpl-pointpicker @ {CANDIDATE_URL}",
        ],
        check=True,
    )
