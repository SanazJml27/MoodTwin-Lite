"""Synthetic wearable-style longitudinal data generator.

The generator is intentionally simple and transparent. It creates plausible-looking
associations between sleep, activity, stress, screen use, HRV, and mood, but it is
not intended to represent real clinical causal effects.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SyntheticConfig:
    n_participants: int = 12
    n_days: int = 180
    start_date: str = "2025-01-01"
    seed: int = 42


def _clip(value: float, lower: float, upper: float) -> float:
    return float(max(lower, min(upper, value)))


def _note_from_day(mood: float, sleep: float, steps: int, stress: int) -> str:
    if mood <= 3.5 and sleep < 6:
        return "low mood after short sleep"
    if stress >= 8:
        return "high stress day"
    if steps > 10000 and mood >= 6.5:
        return "active day with good mood"
    if sleep >= 7.5 and mood >= 7:
        return "restorative sleep and stable mood"
    return "routine day"


def generate_synthetic_data(config: SyntheticConfig | None = None) -> pd.DataFrame:
    """Generate synthetic participant-day data.

    Returns a dataframe with one row per participant per day.
    """

    cfg = config or SyntheticConfig()
    rng = np.random.default_rng(cfg.seed)
    start = date.fromisoformat(cfg.start_date)

    rows: list[dict] = []
    sexes = ["female", "male", "other"]
    risk_groups = ["low", "moderate", "elevated"]

    for p in range(1, cfg.n_participants + 1):
        participant_id = f"MT{p:03d}"
        age = int(rng.integers(19, 67))
        sex = rng.choice(sexes, p=[0.55, 0.4, 0.05])
        risk = rng.choice(risk_groups, p=[0.35, 0.45, 0.20])

        risk_penalty = {"low": 0.0, "moderate": -0.4, "elevated": -0.9}[risk]
        base_mood = rng.normal(6.4, 0.8) + risk_penalty
        base_sleep = rng.normal(7.0, 0.6) + {"low": 0.2, "moderate": 0.0, "elevated": -0.4}[risk]
        base_steps = rng.normal(7600, 1700) + {"low": 700, "moderate": 0, "elevated": -900}[risk]
        base_hrv = rng.normal(48, 12) + {"low": 6, "moderate": 0, "elevated": -7}[risk]
        base_resting_hr = rng.normal(67, 7) + {"low": -2, "moderate": 0, "elevated": 4}[risk]
        adherence_prob = {"low": 0.97, "moderate": 0.91, "elevated": 0.83}[risk]

        previous_mood = _clip(base_mood + rng.normal(0, 0.7), 1, 10)

        for d in range(cfg.n_days):
            current_date = start + timedelta(days=d)
            dow = current_date.weekday()
            weekend = int(dow >= 5)

            seasonal = 0.4 * np.sin(2 * np.pi * d / 90 + rng.normal(0, 0.05))
            weekly_sleep_shift = 0.35 if weekend else -0.05
            weekly_steps_shift = -600 if weekend else 300
            stress_base = 4.5 + (0.8 if not weekend else -0.7) + {"low": -0.5, "moderate": 0.2, "elevated": 1.1}[risk]

            sleep_hours = _clip(rng.normal(base_sleep + weekly_sleep_shift, 0.75), 3.0, 10.5)
            sleep_efficiency = _clip(rng.normal(84 + (sleep_hours - 7) * 2 - (1.2 if risk == "elevated" else 0), 6), 55, 98)
            sleep_midpoint = _clip(rng.normal(3.6 + (0.5 if weekend else 0.0) + (0.3 if risk == "elevated" else 0), 0.7), 1.0, 8.0)

            work_stress = int(round(_clip(rng.normal(stress_base, 1.7), 1, 10)))
            steps = int(round(_clip(rng.normal(base_steps + weekly_steps_shift - 180 * work_stress, 1600), 500, 22000)))
            active_minutes = int(round(_clip(steps / 130 + rng.normal(0, 12), 0, 180)))
            resting_hr = _clip(rng.normal(base_resting_hr + 0.35 * work_stress - 0.08 * (sleep_hours - 7) * 10, 4), 48, 105)
            hrv_rmssd = _clip(rng.normal(base_hrv + 0.004 * (steps - 7000) - 1.2 * (work_stress - 5) + 1.2 * (sleep_hours - 7), 7), 8, 120)
            screen_minutes = int(round(_clip(rng.normal(260 + 25 * work_stress + (55 if weekend else 0), 70), 40, 850)))
            late_screen_minutes = int(round(_clip(rng.normal(50 + 9 * work_stress + max(0, sleep_midpoint - 4) * 25, 30), 0, 300)))
            social_minutes = int(round(_clip(rng.normal(85 + (55 if weekend else 0) - 6 * work_stress, 45), 0, 320)))
            medication_adherence = int(rng.random() < adherence_prob)

            sleep_goodness = -abs(sleep_hours - 7.5) + 1.4
            activity_goodness = np.log1p(steps) - np.log1p(6500)
            hrv_goodness = (hrv_rmssd - 45) / 20
            stress_penalty = -(work_stress - 5) * 0.28
            screen_penalty = -late_screen_minutes / 220
            adherence_effect = 0.25 if medication_adherence else -0.35
            social_effect = (social_minutes - 80) / 260

            expected_mood = (
                0.55 * previous_mood
                + 0.45 * base_mood
                + 0.45 * sleep_goodness
                + 0.35 * activity_goodness
                + 0.25 * hrv_goodness
                + stress_penalty
                + screen_penalty
                + adherence_effect
                + social_effect
                + seasonal
            )
            mood_score = _clip(rng.normal(expected_mood, 0.55), 1, 10)
            anxiety_score = _clip(rng.normal(5.0 + 0.45 * (work_stress - 5) - 0.2 * (sleep_hours - 7) - 0.22 * (mood_score - 5), 0.8), 1, 10)
            energy_score = _clip(rng.normal(5.2 + 0.45 * (sleep_hours - 7) + 0.00012 * (steps - 7000) - 0.28 * (work_stress - 5), 0.8), 1, 10)

            rows.append(
                {
                    "participant_id": participant_id,
                    "date": current_date.isoformat(),
                    "age": age,
                    "sex": sex,
                    "baseline_risk_group": risk,
                    "sleep_hours": round(sleep_hours, 2),
                    "sleep_efficiency": round(sleep_efficiency, 1),
                    "sleep_midpoint_hour": round(sleep_midpoint, 2),
                    "steps": steps,
                    "active_minutes": active_minutes,
                    "resting_hr": round(resting_hr, 1),
                    "hrv_rmssd": round(hrv_rmssd, 1),
                    "screen_minutes": screen_minutes,
                    "late_screen_minutes": late_screen_minutes,
                    "social_minutes": social_minutes,
                    "work_stress": work_stress,
                    "medication_adherence": medication_adherence,
                    "mood_score": round(mood_score, 2),
                    "anxiety_score": round(anxiety_score, 2),
                    "energy_score": round(energy_score, 2),
                    "notes": _note_from_day(mood_score, sleep_hours, steps, work_stress),
                }
            )
            previous_mood = mood_score

    return pd.DataFrame(rows)


def save_synthetic_data(path: str | Path, config: SyntheticConfig | None = None) -> pd.DataFrame:
    """Generate and save synthetic data."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = generate_synthetic_data(config)
    df.to_csv(output_path, index=False)
    return df
