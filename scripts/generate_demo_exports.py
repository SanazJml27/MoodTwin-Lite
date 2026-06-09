"""Generate realistic local wearable-export samples for the portfolio demo.

The tiny CSV/XML files in `examples/` are useful as format templates, but they are
not long enough for a mood-trajectory demo. This script creates 120-day Oura- and
Fitbit-style CSVs, a 90-day Apple Health-style XML, and a 120-day mood diary from
the same synthetic participant timeline.
"""

from __future__ import annotations

from datetime import datetime, time, timedelta, timezone
from pathlib import Path
import sys
from xml.sax.saxutils import escape

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.synthetic_data import SyntheticConfig, generate_synthetic_data

EXAMPLES = ROOT / "examples"


def _first_participant(n_days: int = 120) -> pd.DataFrame:
    df = generate_synthetic_data(SyntheticConfig(n_participants=1, n_days=n_days, start_date="2025-01-01", seed=2026))
    df = df.sort_values("date").reset_index(drop=True)
    return df


def make_mood_diary(df: pd.DataFrame, path: Path) -> None:
    cols = ["date", "mood_score", "anxiety_score", "energy_score", "work_stress", "medication_adherence", "notes"]
    df[cols].to_csv(path, index=False)


def make_oura_csv(df: pd.DataFrame, path: Path) -> None:
    out = pd.DataFrame(
        {
            "day": df["date"],
            "steps": df["steps"],
            # Oura-style exports often represent durations in seconds.
            "total_sleep_duration": (df["sleep_hours"].astype(float) * 3600).round().astype(int),
            "efficiency": df["sleep_efficiency"],
            "sleep_midpoint_hour": df["sleep_midpoint_hour"],
            "average_hrv": df["hrv_rmssd"],
            "lowest_resting_heart_rate": df["resting_hr"],
            "readiness_score": (df["energy_score"].astype(float) * 10).clip(0, 100).round().astype(int),
            "activity_score": (60 + (df["steps"].astype(float) - 6500) / 160).clip(0, 100).round().astype(int),
            "active_minutes": df["active_minutes"],
        }
    )
    out.to_csv(path, index=False)


def make_fitbit_csv(df: pd.DataFrame, path: Path) -> None:
    out = pd.DataFrame(
        {
            "dateTime": df["date"],
            "steps": df["steps"],
            "minutesAsleep": (df["sleep_hours"].astype(float) * 60).round().astype(int),
            "efficiency": df["sleep_efficiency"].round().astype(int),
            "restingHeartRate": df["resting_hr"],
            "active_minutes": df["active_minutes"],
            "sedentaryMinutes": df["screen_minutes"],
        }
    )
    out.to_csv(path, index=False)


def _health_date(day: str, hour: int, minute: int = 0) -> str:
    dt = datetime.combine(pd.to_datetime(day).date(), time(hour=hour, minute=minute), tzinfo=timezone.utc)
    return dt.strftime("%Y-%m-%d %H:%M:%S %z")


def make_apple_health_xml(df: pd.DataFrame, path: Path, n_days: int = 90) -> None:
    rows = []
    for _, row in df.head(n_days).iterrows():
        d = str(row["date"])
        sleep_start = pd.to_datetime(d) - timedelta(days=1) + timedelta(hours=23, minutes=15)
        sleep_end = sleep_start + timedelta(hours=float(row["sleep_hours"]))
        sleep_start_s = sleep_start.replace(tzinfo=timezone.utc).strftime("%Y-%m-%d %H:%M:%S %z")
        sleep_end_s = sleep_end.replace(tzinfo=timezone.utc).strftime("%Y-%m-%d %H:%M:%S %z")
        rows.extend(
            [
                f'<Record type="HKCategoryTypeIdentifierSleepAnalysis" sourceName="MoodTwin Demo" value="HKCategoryValueSleepAnalysisAsleep" startDate="{sleep_start_s}" endDate="{sleep_end_s}"/>',
                f'<Record type="HKQuantityTypeIdentifierStepCount" sourceName="MoodTwin Demo" unit="count" value="{int(row["steps"])}" startDate="{_health_date(d, 8)}" endDate="{_health_date(d, 22)}"/>',
                f'<Record type="HKQuantityTypeIdentifierAppleExerciseTime" sourceName="MoodTwin Demo" unit="min" value="{int(row["active_minutes"])}" startDate="{_health_date(d, 12)}" endDate="{_health_date(d, 13)}"/>',
                f'<Record type="HKQuantityTypeIdentifierRestingHeartRate" sourceName="MoodTwin Demo" unit="count/min" value="{float(row["resting_hr"]):.1f}" startDate="{_health_date(d, 7)}" endDate="{_health_date(d, 7, 5)}"/>',
                f'<Record type="HKQuantityTypeIdentifierHeartRateVariabilitySDNN" sourceName="MoodTwin Demo" unit="ms" value="{float(row["hrv_rmssd"]):.1f}" startDate="{_health_date(d, 7, 10)}" endDate="{_health_date(d, 7, 12)}"/>',
            ]
        )
    xml = "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<HealthData locale=\"en_US\">\n  " + "\n  ".join(rows) + "\n</HealthData>\n"
    path.write_text(xml)


def main() -> None:
    EXAMPLES.mkdir(exist_ok=True)
    df = _first_participant(120)
    make_mood_diary(df, EXAMPLES / "mood_diary_120d_sample.csv")
    make_oura_csv(df, EXAMPLES / "oura_daily_120d_sample.csv")
    make_fitbit_csv(df, EXAMPLES / "fitbit_daily_120d_sample.csv")
    make_apple_health_xml(df, EXAMPLES / "apple_health_90d_sample.xml", n_days=90)
    print("Generated realistic demo exports in examples/:")
    print("- mood_diary_120d_sample.csv")
    print("- oura_daily_120d_sample.csv")
    print("- fitbit_daily_120d_sample.csv")
    print("- apple_health_90d_sample.xml")


if __name__ == "__main__":
    main()
