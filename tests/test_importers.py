from pathlib import Path

import pandas as pd

from src.importers import parse_apple_health_export, parse_fitbit_daily_export, parse_oura_daily_csv
from src.schemas import REQUIRED_COLUMNS

ROOT = Path(__file__).resolve().parents[1]
MOOD_DIARY = pd.read_csv(ROOT / "examples" / "mood_diary_120d_sample.csv")


def test_oura_importer_outputs_schema_for_realistic_sample():
    df, summary = parse_oura_daily_csv(ROOT / "examples" / "oura_daily_120d_sample.csv", mood_diary=MOOD_DIARY)
    assert list(df.columns) == REQUIRED_COLUMNS
    assert len(df) == 120
    assert summary.source == "Oura daily CSV"
    assert summary.start_date == "2025-01-01"
    assert df["mood_score"].nunique() > 20
    assert df["steps"].mean() > 1000


def test_fitbit_importer_outputs_schema_for_realistic_sample():
    df, summary = parse_fitbit_daily_export(ROOT / "examples" / "fitbit_daily_120d_sample.csv", mood_diary=MOOD_DIARY)
    assert list(df.columns) == REQUIRED_COLUMNS
    assert len(df) == 120
    assert summary.source == "Fitbit daily CSV/JSON"
    assert df["sleep_hours"].mean() > 5
    assert df["mood_score"].nunique() > 20


def test_apple_importer_outputs_schema_for_realistic_sample():
    df, summary = parse_apple_health_export(ROOT / "examples" / "apple_health_90d_sample.xml", mood_diary=MOOD_DIARY)
    assert list(df.columns) == REQUIRED_COLUMNS
    assert len(df) == 90
    assert summary.source == "Apple Health export.xml"
    assert summary.start_date == "2025-01-01"
    assert df["steps"].mean() > 1000
    assert df["mood_score"].nunique() > 20


def test_sensor_only_oura_import_generates_nonflat_proxy_mood():
    df, summary = parse_oura_daily_csv(ROOT / "examples" / "oura_daily_120d_sample.csv")
    assert list(df.columns) == REQUIRED_COLUMNS
    assert len(df) == 120
    assert df["mood_score"].nunique() > 10
    assert df["mood_score"].std() > 0.2
    assert any("proxy mood trajectory" in warning for warning in summary.warnings)
