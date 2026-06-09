"""Counterfactual scenario utilities."""

from __future__ import annotations

import pandas as pd

from src.model import forecast_mood


SCENARIOS: dict[str, dict[str, float]] = {
    "Baseline: recent pattern continues": {},
    "Sleep +1 hour, efficiency +5%": {"sleep_hours": 1.0, "sleep_efficiency": 5.0},
    "Activity +2,000 steps/day": {"steps": 2000.0, "active_minutes": 15.0},
    "Less late screen, lower stress": {"late_screen_minutes": -60.0, "screen_minutes": -60.0, "work_stress": -1.0},
    "Combined supportive routine": {
        "sleep_hours": 1.0,
        "sleep_efficiency": 5.0,
        "steps": 2000.0,
        "active_minutes": 15.0,
        "late_screen_minutes": -60.0,
        "screen_minutes": -60.0,
        "work_stress": -1.0,
        "social_minutes": 30.0,
    },
}


def run_counterfactuals(profile_df: pd.DataFrame, model, days: int = 7) -> pd.DataFrame:
    """Run predefined counterfactual scenarios and summarize forecast impact."""

    rows = []
    baseline_mean = None
    for name, scenario in SCENARIOS.items():
        forecast = forecast_mood(profile_df, model, days=days, scenario=scenario)
        mean_mood = float(forecast["predicted_mood"].mean())
        min_mood = float(forecast["predicted_mood"].min())
        if baseline_mean is None:
            baseline_mean = mean_mood
        rows.append(
            {
                "scenario": name,
                "mean_predicted_mood": round(mean_mood, 2),
                "min_predicted_mood": round(min_mood, 2),
                "change_vs_baseline": round(mean_mood - baseline_mean, 2),
            }
        )
    return pd.DataFrame(rows)
