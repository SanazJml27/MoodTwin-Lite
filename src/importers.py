"""Import adapters for wearable and smart-device data.

The project uses a compact daily schema so that very different sources can be
mapped into the same MoodTwin pipeline. These adapters intentionally avoid any
cloud API calls. They work with local exports or CSV/JSON files the user already
has on their computer.
"""

from __future__ import annotations

import json
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from io import BytesIO, StringIO
from pathlib import Path
from typing import BinaryIO, Iterable

import numpy as np
import pandas as pd

from src.schemas import REQUIRED_COLUMNS

DEFAULTS = {
    "age": 35,
    "sex": "other",
    "baseline_risk_group": "moderate",
    "sleep_hours": 7.0,
    "sleep_efficiency": 85.0,
    "sleep_midpoint_hour": 3.5,
    "steps": 6500,
    "active_minutes": 45,
    "resting_hr": 68.0,
    "hrv_rmssd": 45.0,
    "screen_minutes": 240,
    "late_screen_minutes": 45,
    "social_minutes": 90,
    "work_stress": 5,
    "medication_adherence": 1,
    "mood_score": 5.5,
    "anxiety_score": 5.0,
    "energy_score": 5.5,
    "notes": "imported wearable row; missing fields filled by defaults",
}

MOOD_COLUMNS = ["mood_score", "anxiety_score", "energy_score", "work_stress", "medication_adherence", "notes"]


@dataclass(frozen=True)
class ImportSummary:
    """Small user-facing summary returned by import helpers."""

    source: str
    rows: int
    participant_count: int
    start_date: str
    end_date: str
    warnings: list[str]


def _read_csv_like(file_or_path) -> pd.DataFrame:
    if isinstance(file_or_path, pd.DataFrame):
        return file_or_path.copy()
    if isinstance(file_or_path, (str, Path)):
        return pd.read_csv(file_or_path)
    # Streamlit UploadedFile exposes bytes through read(). Reset handled by caller.
    content = file_or_path.read()
    if isinstance(content, bytes):
        content = content.decode("utf-8")
    return pd.read_csv(StringIO(content))


def _date_series(values) -> pd.Series:
    parsed = pd.to_datetime(values, errors="coerce", utc=True)
    return parsed.dt.tz_convert(None).dt.date.astype("string")


def _first_existing_column(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    lower_map = {str(c).lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]
    return None


def _column_contains(df: pd.DataFrame, *needles: str) -> str | None:
    lowered = [(str(c).lower(), c) for c in df.columns]
    for low, original in lowered:
        if all(n.lower() in low for n in needles):
            return original
    return None


def _safe_numeric(df: pd.DataFrame, column: str | None, default: float = np.nan) -> pd.Series:
    if column is None or column not in df.columns:
        return pd.Series(default, index=df.index, dtype="float64")
    return pd.to_numeric(df[column], errors="coerce")


def _scale_to_1_10(series: pd.Series, min_value: float | None = None, max_value: float | None = None) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    if min_value is None:
        min_value = float(s.quantile(0.05)) if s.notna().any() else 0.0
    if max_value is None:
        max_value = float(s.quantile(0.95)) if s.notna().any() else 100.0
    if math.isclose(max_value, min_value):
        return pd.Series(5.5, index=s.index)
    scaled = 1 + 9 * (s - min_value) / (max_value - min_value)
    return scaled.clip(1, 10)


def ensure_moodtwin_schema(
    df: pd.DataFrame,
    *,
    participant_id: str = "USER001",
    age: int = 35,
    sex: str = "other",
    baseline_risk_group: str = "moderate",
) -> tuple[pd.DataFrame, list[str]]:
    """Fill and order the canonical MoodTwin schema.

    This makes personal wearable exports usable even when they do not include
    self-reported mood fields. The app warns users when mood was imputed, because
    a sensor-only import cannot support meaningful supervised mood forecasting by
    itself.
    """

    out = df.copy()
    warnings: list[str] = []

    if "date" not in out.columns:
        raise ValueError("Imported data must contain a date column after conversion.")
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.date.astype("string")
    out = out.dropna(subset=["date"]).copy()

    if "participant_id" not in out.columns:
        out["participant_id"] = participant_id
    out["participant_id"] = out["participant_id"].fillna(participant_id).astype(str)

    for col, value in {
        "age": age,
        "sex": sex,
        "baseline_risk_group": baseline_risk_group,
    }.items():
        if col not in out.columns:
            out[col] = value
        out[col] = out[col].fillna(value)

    for col, default in DEFAULTS.items():
        if col not in out.columns:
            out[col] = default
            if col in MOOD_COLUMNS:
                warnings.append(f"{col} was not present and was filled with a neutral default.")
        elif col != "notes":
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(default)
        else:
            out[col] = out[col].fillna(default).astype(str)

    if out["mood_score"].nunique(dropna=True) <= 2:
        warnings.append(
            "Mood labels appear missing or nearly constant. For a real personal twin, merge a daily mood diary CSV."
        )

    # Sensible clipping to keep the downstream Pydantic/schema assumptions valid.
    clipping = {
        "sleep_hours": (0, 16),
        "sleep_efficiency": (0, 100),
        "sleep_midpoint_hour": (0, 24),
        "steps": (0, 60000),
        "active_minutes": (0, 600),
        "resting_hr": (30, 140),
        "hrv_rmssd": (1, 250),
        "screen_minutes": (0, 1440),
        "late_screen_minutes": (0, 600),
        "social_minutes": (0, 600),
        "work_stress": (1, 10),
        "medication_adherence": (0, 1),
        "mood_score": (1, 10),
        "anxiety_score": (1, 10),
        "energy_score": (1, 10),
    }
    for col, bounds in clipping.items():
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(DEFAULTS[col]).clip(*bounds)

    int_columns = ["age", "steps", "active_minutes", "screen_minutes", "late_screen_minutes", "social_minutes", "work_stress", "medication_adherence"]
    for col in int_columns:
        out[col] = out[col].round().astype(int)

    out = out[REQUIRED_COLUMNS].sort_values(["participant_id", "date"]).reset_index(drop=True)
    return out, sorted(set(warnings))


def merge_mood_diary(sensor_df: pd.DataFrame, mood_diary: pd.DataFrame | None) -> tuple[pd.DataFrame, list[str]]:
    """Merge optional daily mood diary values into a sensor dataframe.

    Expected diary columns: date and any of mood_score, anxiety_score,
    energy_score, work_stress, medication_adherence, notes.
    """

    if mood_diary is None or mood_diary.empty:
        return sensor_df, []

    diary = mood_diary.copy()
    if "date" not in diary.columns:
        raise ValueError("Mood diary must contain a date column.")
    diary["date"] = pd.to_datetime(diary["date"], errors="coerce").dt.date.astype("string")
    keep = ["date"] + [c for c in MOOD_COLUMNS if c in diary.columns]
    diary = diary[keep].dropna(subset=["date"]).drop_duplicates("date", keep="last")

    out = sensor_df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.date.astype("string")
    out = out.merge(diary, on="date", how="left", suffixes=("", "_diary"))

    for col in MOOD_COLUMNS:
        diary_col = f"{col}_diary"
        if diary_col in out.columns:
            if col in out.columns:
                out[col] = out[col].where(out[diary_col].isna(), out[diary_col])
            else:
                out[col] = out[diary_col]
            out = out.drop(columns=[diary_col])

    return out, ["Merged optional daily mood diary values into wearable data."]


def parse_apple_health_export(
    xml_file: str | Path | BinaryIO | BytesIO,
    *,
    participant_id: str = "APPLE001",
    mood_diary: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, ImportSummary]:
    """Convert Apple Health export.xml records into daily MoodTwin rows.

    The parser uses common HealthKit identifiers and creates daily summaries.
    It is intentionally conservative: missing fields are filled by defaults and
    reported as warnings.
    """

    if isinstance(xml_file, (str, Path)):
        tree = ET.parse(xml_file)
    else:
        content = xml_file.read()
        if isinstance(content, str):
            content = content.encode("utf-8")
        tree = ET.parse(BytesIO(content))

    records: list[dict] = []
    sleep_intervals: list[dict] = []
    for rec in tree.getroot().iter("Record"):
        attrs = rec.attrib
        rtype = attrs.get("type", "")
        start = attrs.get("startDate")
        end = attrs.get("endDate")
        value = attrs.get("value")
        if not start:
            continue
        if "SleepAnalysis" in rtype:
            # Apple values have changed over time; anything with Asleep is counted.
            if value and "asleep" in str(value).lower():
                sleep_intervals.append({"start": start, "end": end or start})
            continue
        records.append({"type": rtype, "startDate": start, "endDate": end, "value": value, "unit": attrs.get("unit")})

    if not records and not sleep_intervals:
        raise ValueError("No Apple Health Record elements were found in the XML file.")

    daily = pd.DataFrame()
    if records:
        rec_df = pd.DataFrame(records)
        rec_df["date"] = _date_series(rec_df["startDate"])
        rec_df["numeric_value"] = pd.to_numeric(rec_df["value"], errors="coerce")

        def summarize_contains(keyword: str, agg: str = "sum") -> pd.Series:
            subset = rec_df[rec_df["type"].str.contains(keyword, case=False, na=False)]
            if subset.empty:
                return pd.Series(dtype="float64")
            grouped = subset.groupby("date")["numeric_value"]
            return grouped.sum() if agg == "sum" else grouped.mean()

        pieces = [pd.DataFrame(index=sorted(rec_df["date"].dropna().unique()))]
        pieces[0].index.name = "date"
        pieces[0]["steps"] = summarize_contains("StepCount", "sum")
        pieces[0]["active_minutes"] = summarize_contains("ExerciseTime", "sum")
        pieces[0]["resting_hr"] = summarize_contains("RestingHeartRate", "mean")
        # Apple HRV is commonly SDNN, but MoodTwin uses one HRV slot. We keep the slot name and explain in notes.
        pieces[0]["hrv_rmssd"] = summarize_contains("HeartRateVariability", "mean")
        daily = pieces[0].reset_index()

    if sleep_intervals:
        sleep_df = pd.DataFrame(sleep_intervals)
        sleep_df["start_dt"] = pd.to_datetime(sleep_df["start"], errors="coerce", utc=True)
        sleep_df["end_dt"] = pd.to_datetime(sleep_df["end"], errors="coerce", utc=True)
        sleep_df = sleep_df.dropna(subset=["start_dt", "end_dt"])
        # Assign overnight sleep to the wake-up day, which is usually what users expect in a daily mood timeline.
        sleep_df["date"] = sleep_df["end_dt"].dt.tz_convert(None).dt.date.astype("string")
        sleep_df["hours"] = (sleep_df["end_dt"] - sleep_df["start_dt"]).dt.total_seconds() / 3600

        midpoint = sleep_df["start_dt"] + (sleep_df["end_dt"] - sleep_df["start_dt"]) / 2
        sleep_df["midpoint_hour"] = midpoint.dt.hour + midpoint.dt.minute / 60
        sleep_daily = (
            sleep_df.groupby("date", as_index=False)
            .agg(sleep_hours=("hours", "sum"), sleep_midpoint_hour=("midpoint_hour", "mean"))
        )
        daily = sleep_daily if daily.empty else daily.merge(sleep_daily, on="date", how="outer")

    daily["participant_id"] = participant_id
    daily["notes"] = "imported from Apple Health export.xml; Apple HRV may represent SDNN, mapped into hrv_rmssd slot"
    daily, merge_warnings = merge_mood_diary(daily, mood_diary)
    out, warnings = ensure_moodtwin_schema(daily, participant_id=participant_id)
    warnings.extend(merge_warnings)
    return out, build_import_summary(out, "Apple Health export.xml", warnings)


def parse_oura_daily_csv(
    csv_file,
    *,
    participant_id: str = "OURA001",
    mood_diary: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, ImportSummary]:
    """Convert an Oura-like daily CSV into MoodTwin rows.

    Accepts common Oura export/API-derived column names such as day, date,
    summary_date, steps, total_sleep_duration, efficiency, average_hrv,
    lowest_resting_heart_rate, readiness_score, and activity_score.
    """

    raw = _read_csv_like(csv_file)
    date_col = _first_existing_column(raw, ["date", "day", "summary_date", "calendar_date"])
    if date_col is None:
        raise ValueError("Oura CSV must include a date/day/summary_date column.")

    out = pd.DataFrame({"date": _date_series(raw[date_col]), "participant_id": participant_id})
    out["steps"] = _safe_numeric(raw, _first_existing_column(raw, ["steps", "total_steps"]))
    out["active_minutes"] = _safe_numeric(raw, _first_existing_column(raw, ["active_minutes", "high_activity_time", "medium_activity_time"]), 45)
    sleep_col = _first_existing_column(raw, ["sleep_hours", "total_sleep_duration", "total_sleep_time", "duration"])
    sleep_value = _safe_numeric(raw, sleep_col, 7.0)
    # Oura exports often use seconds. Convert large values to hours.
    out["sleep_hours"] = np.where(sleep_value > 24, sleep_value / 3600, sleep_value)
    out["sleep_efficiency"] = _safe_numeric(raw, _first_existing_column(raw, ["sleep_efficiency", "efficiency"]), 85)
    out["sleep_midpoint_hour"] = _safe_numeric(raw, _first_existing_column(raw, ["sleep_midpoint_hour", "midpoint_time"]), 3.5)
    out["resting_hr"] = _safe_numeric(raw, _first_existing_column(raw, ["resting_hr", "lowest_resting_heart_rate", "average_heart_rate"]), 68)
    out["hrv_rmssd"] = _safe_numeric(raw, _first_existing_column(raw, ["hrv_rmssd", "average_hrv", "hrv"]), 45)
    readiness = _safe_numeric(raw, _first_existing_column(raw, ["readiness_score", "score", "activity_score"]), np.nan)
    if readiness.notna().any():
        out["energy_score"] = _scale_to_1_10(readiness, 0, 100)
    out["notes"] = "imported from Oura-like daily CSV"
    out, merge_warnings = merge_mood_diary(out, mood_diary)
    out, warnings = ensure_moodtwin_schema(out, participant_id=participant_id)
    warnings.extend(merge_warnings)
    return out, build_import_summary(out, "Oura daily CSV", warnings)


def parse_fitbit_daily_export(
    file_or_path,
    *,
    participant_id: str = "FITBIT001",
    mood_diary: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, ImportSummary]:
    """Convert a Fitbit-like CSV or JSON file into MoodTwin rows.

    The CSV path accepts columns such as date/dateTime, steps, active_minutes,
    minutesAsleep/sleep_minutes, restingHeartRate, caloriesOut, and sedentaryMinutes.
    JSON support handles simple lists of daily dictionaries, plus common Fitbit
    API shapes with activities-heart and activities-steps arrays.
    """

    if isinstance(file_or_path, (str, Path)):
        path = Path(file_or_path)
        text = path.read_text()
        suffix = path.suffix.lower()
    else:
        raw = file_or_path.read()
        text = raw.decode("utf-8") if isinstance(raw, bytes) else raw
        suffix = ".json" if text.lstrip().startswith(("{", "[")) else ".csv"

    if suffix == ".json":
        payload = json.loads(text)
        rows: list[dict] = []
        if isinstance(payload, list):
            rows = payload
        elif isinstance(payload, dict):
            # Flatten a few common Fitbit Web API response fragments.
            by_date: dict[str, dict] = {}
            for item in payload.get("activities-steps", []):
                d = item.get("dateTime")
                by_date.setdefault(d, {})["steps"] = item.get("value")
            for item in payload.get("activities-heart", []):
                d = item.get("dateTime")
                value = item.get("value", {}) if isinstance(item.get("value"), dict) else {}
                by_date.setdefault(d, {})["restingHeartRate"] = value.get("restingHeartRate")
            for item in payload.get("sleep", []):
                d = item.get("dateOfSleep") or item.get("dateTime")
                by_date.setdefault(d, {})["minutesAsleep"] = item.get("minutesAsleep")
                by_date.setdefault(d, {})["efficiency"] = item.get("efficiency")
            rows = [{"dateTime": k, **v} for k, v in by_date.items() if k]
        raw = pd.DataFrame(rows)
    else:
        raw = pd.read_csv(StringIO(text))

    date_col = _first_existing_column(raw, ["date", "dateTime", "dateOfSleep", "day"])
    if date_col is None:
        raise ValueError("Fitbit file must include a date/dateTime/dateOfSleep column.")

    out = pd.DataFrame({"date": _date_series(raw[date_col]), "participant_id": participant_id})
    out["steps"] = _safe_numeric(raw, _first_existing_column(raw, ["steps", "activities_steps"]), 6500)
    out["active_minutes"] = _safe_numeric(raw, _first_existing_column(raw, ["active_minutes", "fairlyActiveMinutes", "veryActiveMinutes", "lightlyActiveMinutes"]), 45)
    sleep_min = _safe_numeric(raw, _first_existing_column(raw, ["minutesAsleep", "sleep_minutes", "minutes_asleep"]), np.nan)
    out["sleep_hours"] = sleep_min / 60
    out["sleep_efficiency"] = _safe_numeric(raw, _first_existing_column(raw, ["efficiency", "sleep_efficiency"]), 85)
    out["resting_hr"] = _safe_numeric(raw, _first_existing_column(raw, ["restingHeartRate", "resting_hr", "resting_heart_rate"]), 68)
    out["screen_minutes"] = _safe_numeric(raw, _first_existing_column(raw, ["screen_minutes", "sedentaryMinutes", "sedentary_minutes"]), 240)
    out["notes"] = "imported from Fitbit-like daily CSV/JSON"
    out, merge_warnings = merge_mood_diary(out, mood_diary)
    out, warnings = ensure_moodtwin_schema(out, participant_id=participant_id)
    warnings.extend(merge_warnings)
    return out, build_import_summary(out, "Fitbit daily CSV/JSON", warnings)


def build_import_summary(df: pd.DataFrame, source: str, warnings: list[str] | None = None) -> ImportSummary:
    dates = pd.to_datetime(df["date"], errors="coerce")
    return ImportSummary(
        source=source,
        rows=int(len(df)),
        participant_count=int(df["participant_id"].nunique()),
        start_date=str(dates.min().date()) if dates.notna().any() else "unknown",
        end_date=str(dates.max().date()) if dates.notna().any() else "unknown",
        warnings=sorted(set(warnings or [])),
    )
