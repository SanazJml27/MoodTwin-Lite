"""Feature engineering utilities for MoodTwin-Lite."""

from __future__ import annotations

import numpy as np
import pandas as pd

PREDICTOR_COLUMNS = [
    "day_of_week_sin",
    "day_of_week_cos",
    "sleep_hours",
    "sleep_efficiency",
    "sleep_midpoint_hour",
    "steps",
    "active_minutes",
    "resting_hr",
    "hrv_rmssd",
    "screen_minutes",
    "late_screen_minutes",
    "social_minutes",
    "work_stress",
    "medication_adherence",
    "mood_lag1",
    "mood_rolling3",
    "mood_rolling7",
    "sleep_rolling7",
    "steps_rolling7",
    "hrv_rolling7",
    "stress_rolling7",
]

TARGET_COLUMN = "target_next_day_mood"


def clean_input_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize dates, sort rows, and coerce numeric columns."""

    cleaned = df.copy()
    cleaned["date"] = pd.to_datetime(cleaned["date"])
    cleaned = cleaned.sort_values(["participant_id", "date"]).reset_index(drop=True)

    numeric_columns = [
        "age",
        "sleep_hours",
        "sleep_efficiency",
        "sleep_midpoint_hour",
        "steps",
        "active_minutes",
        "resting_hr",
        "hrv_rmssd",
        "screen_minutes",
        "late_screen_minutes",
        "social_minutes",
        "work_stress",
        "medication_adherence",
        "mood_score",
        "anxiety_score",
        "energy_score",
    ]
    for col in numeric_columns:
        cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")

    return cleaned


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add cyclical day-of-week features."""

    out = df.copy()
    dow = out["date"].dt.dayofweek.astype(float)
    out["day_of_week_sin"] = np.sin(2 * np.pi * dow / 7)
    out["day_of_week_cos"] = np.cos(2 * np.pi * dow / 7)
    return out


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add lagged and rolling features per participant."""

    out = add_time_features(clean_input_dataframe(df))
    grouped = out.groupby("participant_id", group_keys=False)

    out["mood_lag1"] = grouped["mood_score"].shift(1)
    out["mood_rolling3"] = grouped["mood_score"].transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    out["mood_rolling7"] = grouped["mood_score"].transform(lambda s: s.shift(1).rolling(7, min_periods=1).mean())
    out["sleep_rolling7"] = grouped["sleep_hours"].transform(lambda s: s.shift(1).rolling(7, min_periods=1).mean())
    out["steps_rolling7"] = grouped["steps"].transform(lambda s: s.shift(1).rolling(7, min_periods=1).mean())
    out["hrv_rolling7"] = grouped["hrv_rmssd"].transform(lambda s: s.shift(1).rolling(7, min_periods=1).mean())
    out["stress_rolling7"] = grouped["work_stress"].transform(lambda s: s.shift(1).rolling(7, min_periods=1).mean())

    out[TARGET_COLUMN] = grouped["mood_score"].shift(-1)
    return out


def prepare_supervised_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Return X, y, and feature dataframe for supervised next-day mood prediction."""

    featured = add_rolling_features(df)
    supervised = featured.dropna(subset=PREDICTOR_COLUMNS + [TARGET_COLUMN]).copy()
    X = supervised[PREDICTOR_COLUMNS]
    y = supervised[TARGET_COLUMN]
    return X, y, supervised


def validate_required_columns(df: pd.DataFrame, required_columns: list[str]) -> list[str]:
    """Return missing columns from a dataframe."""

    return [col for col in required_columns if col not in df.columns]
