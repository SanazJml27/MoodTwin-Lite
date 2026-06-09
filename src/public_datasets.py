"""Adapters for public digital-phenotyping datasets.

The main supported path is the Dryad IFH Affect dataset:
"physiological and emotional assessment of college students using wearable and
mobile devices during the 2020 COVID-19 lockdown". Users download/unzip the
public dataset separately, then run scripts/adapt_ifh_affect.py locally.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.importers import _first_existing_column, _safe_numeric, _scale_to_1_10, ensure_moodtwin_schema

POSITIVE_AFFECT = [
    "Active",
    "Alert",
    "Attentive",
    "Determined",
    "Enthusiastic",
    "Excited",
    "Inspired",
    "Interested",
    "Proud",
    "Strong",
]
NEGATIVE_AFFECT = [
    "Afraid",
    "Ashamed",
    "Distressed",
    "Guilty",
    "Hostile",
    "Irritable",
    "Jittery",
    "Nervous",
    "Scared",
    "Upset",
]


def _parse_epoch_or_date(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().mean() > 0.5:
        # Dataset documentation describes millisecond epoch timestamps for EMA.
        dt = pd.to_datetime(numeric, unit="ms", errors="coerce", utc=True)
    else:
        dt = pd.to_datetime(series, errors="coerce", utc=True)
    return dt.dt.tz_convert(None).dt.date.astype("string")


def _date_col(df: pd.DataFrame) -> str | None:
    return _first_existing_column(df, ["date", "day", "summary_date", "submission_timestamp", "timestamp", "start_timestamp"])


def _read_optional_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _daily_ema_to_mood(ema_df: pd.DataFrame, participant_id: str) -> pd.DataFrame:
    if ema_df.empty:
        return pd.DataFrame(columns=["date", "participant_id", "mood_score", "anxiety_score", "energy_score", "work_stress"])
    date_col = _date_col(ema_df)
    if date_col is None:
        return pd.DataFrame(columns=["date", "participant_id"])
    out = pd.DataFrame({"date": _parse_epoch_or_date(ema_df[date_col]), "participant_id": participant_id})
    positive_cols = [c for c in POSITIVE_AFFECT if c in ema_df.columns]
    negative_cols = [c for c in NEGATIVE_AFFECT if c in ema_df.columns]
    if positive_cols:
        pos = ema_df[positive_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)
    else:
        pos = pd.Series(50, index=ema_df.index)
    if negative_cols:
        neg = ema_df[negative_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)
    else:
        neg = pd.Series(35, index=ema_df.index)
    balance = (pos - neg + 100) / 2  # maps -100..100 to 0..100
    out["mood_score"] = _scale_to_1_10(balance, 0, 100)
    out["anxiety_score"] = _scale_to_1_10(neg, 0, 100)
    out["energy_score"] = _scale_to_1_10(ema_df["Active"] if "Active" in ema_df.columns else pos, 0, 100)
    stress_col = _first_existing_column(ema_df, ["Stress", "stressed", "Distressed", "Nervous"])
    out["work_stress"] = _scale_to_1_10(ema_df[stress_col] if stress_col else neg, 0, 100)
    out["notes"] = "mood estimated from daily EMA positive/negative affect balance"
    return out.groupby(["participant_id", "date"], as_index=False).mean(numeric_only=True).merge(
        out[["participant_id", "date", "notes"]].drop_duplicates(), on=["participant_id", "date"], how="left"
    )


def _oura_activity_to_daily(activity_df: pd.DataFrame, participant_id: str) -> pd.DataFrame:
    if activity_df.empty:
        return pd.DataFrame(columns=["date", "participant_id"])
    date_col = _date_col(activity_df)
    if date_col is None:
        return pd.DataFrame(columns=["date", "participant_id"])
    out = pd.DataFrame({"date": _parse_epoch_or_date(activity_df[date_col]), "participant_id": participant_id})
    out["steps"] = _safe_numeric(activity_df, _first_existing_column(activity_df, ["steps", "total_steps"]), np.nan)
    out["active_minutes"] = _safe_numeric(
        activity_df,
        _first_existing_column(activity_df, ["active_minutes", "high_activity_time", "medium_activity_time", "low_activity_time"]),
        np.nan,
    )
    return out.groupby(["participant_id", "date"], as_index=False).mean(numeric_only=True)


def _oura_sleep_to_daily(sleep_df: pd.DataFrame, participant_id: str) -> pd.DataFrame:
    if sleep_df.empty:
        return pd.DataFrame(columns=["date", "participant_id"])
    date_col = _date_col(sleep_df)
    if date_col is None:
        return pd.DataFrame(columns=["date", "participant_id"])
    out = pd.DataFrame({"date": _parse_epoch_or_date(sleep_df[date_col]), "participant_id": participant_id})
    sleep_col = _first_existing_column(sleep_df, ["sleep_hours", "total_sleep_duration", "total", "duration"])
    sleep = _safe_numeric(sleep_df, sleep_col, np.nan)
    out["sleep_hours"] = np.where(sleep > 24, sleep / 3600, sleep)
    out["sleep_efficiency"] = _safe_numeric(sleep_df, _first_existing_column(sleep_df, ["sleep_efficiency", "efficiency"]), np.nan)
    midpoint_col = _first_existing_column(sleep_df, ["midpoint_time", "sleep_midpoint_hour"])
    out["sleep_midpoint_hour"] = _safe_numeric(sleep_df, midpoint_col, np.nan)
    return out.groupby(["participant_id", "date"], as_index=False).mean(numeric_only=True)


def _oura_readiness_to_daily(readiness_df: pd.DataFrame, participant_id: str) -> pd.DataFrame:
    if readiness_df.empty:
        return pd.DataFrame(columns=["date", "participant_id"])
    date_col = _date_col(readiness_df)
    if date_col is None:
        return pd.DataFrame(columns=["date", "participant_id"])
    out = pd.DataFrame({"date": _parse_epoch_or_date(readiness_df[date_col]), "participant_id": participant_id})
    out["hrv_rmssd"] = _safe_numeric(readiness_df, _first_existing_column(readiness_df, ["average_hrv", "hrv", "hrv_rmssd"]), np.nan)
    out["resting_hr"] = _safe_numeric(readiness_df, _first_existing_column(readiness_df, ["resting_hr", "lowest_resting_heart_rate", "average_heart_rate"]), np.nan)
    score = _safe_numeric(readiness_df, _first_existing_column(readiness_df, ["readiness_score", "score"]), np.nan)
    if score.notna().any():
        out["energy_score"] = _scale_to_1_10(score, 0, 100)
    return out.groupby(["participant_id", "date"], as_index=False).mean(numeric_only=True)


def adapt_ifh_affect_dataset(root_dir: str | Path, output_path: str | Path | None = None) -> pd.DataFrame:
    """Adapt the public IFH Affect Dryad dataset directory into MoodTwin rows.

    Expected structure after unzipping ifh_affect.zip:
        par_1/ema/daily.csv
        par_1/oura/activity.csv
        par_1/oura/sleep.csv
        par_1/oura/readiness.csv
        ...

    The target mood is not a diagnosis score; it is a demo-friendly 1-10 mood
    proxy derived from daily EMA positive-vs-negative affect balance.
    """

    root = Path(root_dir)
    participant_dirs = sorted(p for p in root.glob("par_*") if p.is_dir())
    if not participant_dirs:
        raise FileNotFoundError(f"No par_* participant folders found under {root}")

    all_rows: list[pd.DataFrame] = []
    for participant_dir in participant_dirs:
        pid = participant_dir.name.replace("par_", "IFH")
        ema = _daily_ema_to_mood(_read_optional_csv(participant_dir / "ema" / "daily.csv"), pid)
        act = _oura_activity_to_daily(_read_optional_csv(participant_dir / "oura" / "activity.csv"), pid)
        sleep = _oura_sleep_to_daily(_read_optional_csv(participant_dir / "oura" / "sleep.csv"), pid)
        readiness = _oura_readiness_to_daily(_read_optional_csv(participant_dir / "oura" / "readiness.csv"), pid)

        merged = ema
        for piece in [act, sleep, readiness]:
            if merged.empty:
                merged = piece
            elif not piece.empty:
                merged = merged.merge(piece, on=["participant_id", "date"], how="outer")
        if not merged.empty:
            merged["notes"] = merged.get("notes", "public IFH Affect dataset row")
            all_rows.append(merged)

    if not all_rows:
        raise ValueError("No usable IFH Affect rows were created from the dataset directory.")

    raw = pd.concat(all_rows, ignore_index=True)
    out, _warnings = ensure_moodtwin_schema(raw, participant_id="IFH001", age=21, sex="other", baseline_risk_group="moderate")
    out["notes"] = out["notes"].replace("imported wearable row; missing fields filled by defaults", "public IFH Affect dataset row")

    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(path, index=False)
    return out
