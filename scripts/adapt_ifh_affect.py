#!/usr/bin/env python3
"""Adapt the public Dryad IFH Affect dataset into MoodTwin format.

Download the dataset from Dryad, unzip ifh_affect.zip locally, then run:

  python scripts/adapt_ifh_affect.py --root /path/to/ifh_affect --output data/ifh_affect_moodtwin.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.public_datasets import adapt_ifh_affect_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Adapt IFH Affect Dryad dataset into MoodTwin-Lite CSV format.")
    parser.add_argument("--root", required=True, help="Path to unzipped ifh_affect folder containing par_* folders.")
    parser.add_argument("--output", default="data/ifh_affect_moodtwin.csv")
    args = parser.parse_args()

    df = adapt_ifh_affect_dataset(args.root, args.output)
    print(f"Saved {len(df)} rows for {df['participant_id'].nunique()} participants to {Path(args.output)}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")


if __name__ == "__main__":
    main()
