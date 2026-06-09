"""Risk scoring helpers."""

from __future__ import annotations

import pandas as pd


def assess_deterioration_risk(history_df: pd.DataFrame, forecast_df: pd.DataFrame) -> dict[str, object]:
    """Assess near-term mood deterioration risk from history and forecast.

    This is a transparent heuristic, not a clinical risk model.
    """

    recent = history_df.sort_values("date").tail(7)
    recent_mean = float(recent["mood_score"].mean())
    forecast_mean = float(forecast_df["predicted_mood"].mean())
    forecast_min = float(forecast_df["predicted_mood"].min())
    change = forecast_mean - recent_mean

    if forecast_min < 3.5 or change <= -1.25:
        level = "elevated"
    elif forecast_min < 4.5 or change <= -0.65:
        level = "moderate"
    else:
        level = "low"

    drivers = []
    recent_sleep = float(recent["sleep_hours"].mean())
    recent_steps = float(recent["steps"].mean())
    recent_stress = float(recent["work_stress"].mean())
    recent_late_screen = float(recent["late_screen_minutes"].mean())
    recent_hrv = float(recent["hrv_rmssd"].mean())

    if recent_sleep < 6.4:
        drivers.append("recent sleep duration is below the preferred range")
    if recent_steps < 5000:
        drivers.append("recent activity level is low")
    if recent_stress >= 7:
        drivers.append("recent stress score is high")
    if recent_late_screen > 110:
        drivers.append("late-night screen exposure is high")
    if recent_hrv < 35:
        drivers.append("recent HRV is low relative to typical healthy recovery patterns")
    if not drivers:
        drivers.append("no single strong driver; forecast is based on the combined recent pattern")

    return {
        "risk_level": level,
        "recent_7d_mood_mean": round(recent_mean, 2),
        "forecast_7d_mood_mean": round(forecast_mean, 2),
        "forecast_min": round(forecast_min, 2),
        "forecast_change": round(change, 2),
        "drivers": drivers,
    }
