#!/usr/bin/env python3
"""Convert local wearable exports into MoodTwin schema CSV.

Examples:
  python scripts/convert_wearable_export.py --source apple --input export.xml --mood-diary mood.csv --output data/my_apple_moodtwin.csv
  python scripts/convert_wearable_export.py --source oura --input oura_daily.csv --output data/my_oura_moodtwin.csv
  python scripts/convert_wearable_export.py --source fitbit --input fitbit_daily.json --output data/my_fitbit_moodtwin.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.importers import parse_apple_health_export, parse_fitbit_daily_export, parse_oura_daily_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert wearable exports into MoodTwin-Lite CSV format.")
    parser.add_argument("--source", choices=["apple", "oura", "fitbit"], required=True)
    parser.add_argument("--input", required=True, help="Path to export.xml, Oura CSV, Fitbit CSV, or Fitbit JSON.")
    parser.add_argument("--output", default="data/imported_moodtwin.csv")
    parser.add_argument("--participant-id", default=None)
    parser.add_argument("--mood-diary", default=None, help="Optional CSV with date,mood_score,anxiety_score,energy_score,etc.")
    args = parser.parse_args()

    mood_diary = pd.read_csv(args.mood_diary) if args.mood_diary else None
    participant_id = args.participant_id or {"apple": "APPLE001", "oura": "OURA001", "fitbit": "FITBIT001"}[args.source]

    if args.source == "apple":
        df, summary = parse_apple_health_export(args.input, participant_id=participant_id, mood_diary=mood_diary)
    elif args.source == "oura":
        df, summary = parse_oura_daily_csv(args.input, participant_id=participant_id, mood_diary=mood_diary)
    else:
        df, summary = parse_fitbit_daily_export(args.input, participant_id=participant_id, mood_diary=mood_diary)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)

    print(f"Saved {len(df)} MoodTwin rows to {output}")
    print(summary)
    if summary.warnings:
        print("Warnings:")
        for warning in summary.warnings:
            print(f"- {warning}")


if __name__ == "__main__":
    main()
