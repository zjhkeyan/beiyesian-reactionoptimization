#!/usr/bin/env python3
"""Run the locked final optimizer on both datasets and all official seeds."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


OFFICIAL_SEEDS = tuple(range(100, 2001, 100))
CODE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Directory containing Suzuki/ and Buchwald_sub4/.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=CODE_ROOT.parent / "results/experimental_investigations/sota_search",
    )
    parser.add_argument("--jobs", type=int, default=6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    phases = (
        ("refinement_screening", OFFICIAL_SEEDS[:3]),
        ("refinement_confirmation", OFFICIAL_SEEDS[3:]),
    )
    for phase, seeds in phases:
        command = [
            sys.executable,
            str(CODE_ROOT / "experiments/run_sota_search.py"),
            "--dataset-root",
            str(args.dataset_root.resolve()),
            "--output-dir",
            str(args.output_dir.resolve()),
            "--phase",
            phase,
            "--datasets",
            "Buchwald_sub4",
            "Suzuki",
            "--variants",
            "gp_interaction_gp_warmstart_logei",
            "--seeds",
            *(str(seed) for seed in seeds),
            "--budget",
            "40",
            "--jobs",
            str(args.jobs),
        ]
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
