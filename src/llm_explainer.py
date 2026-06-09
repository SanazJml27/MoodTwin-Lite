"""Explanation layer for MoodTwin-Lite.

The deterministic explanation works fully offline. The LLM prompt builder is used
only when a user explicitly enables an optional LLM provider.
"""

from __future__ import annotations

import pandas as pd


def explain_forecast(history_df: pd.DataFrame, forecast_df: pd.DataFrame, risk: dict[str, object]) -> str:
    """Create a deterministic local explanation with no external API call."""

    recent = history_df.sort_values("date").tail(7)
    sleep = recent["sleep_hours"].mean()
    steps = recent["steps"].mean()
    stress = recent["work_stress"].mean()
    late_screen = recent["late_screen_minutes"].mean()
    hrv = recent["hrv_rmssd"].mean()

    lines = [
        f"The model estimates a {risk['risk_level']} near-term deterioration risk.",
        f"Recent 7-day mean mood was {risk['recent_7d_mood_mean']}/10, while the forecast 7-day mean is {risk['forecast_7d_mood_mean']}/10.",
        f"The lowest predicted value in the forecast window is {risk['forecast_min']}/10.",
        "",
        "Main recent markers:",
        f"- Sleep: {sleep:.1f} hours/night on average.",
        f"- Activity: {steps:.0f} steps/day on average.",
        f"- Stress: {stress:.1f}/10 on average.",
        f"- Late screen exposure: {late_screen:.0f} minutes/day on average.",
        f"- HRV RMSSD: {hrv:.1f} ms on average.",
        "",
        "Interpretation:",
    ]

    for driver in risk["drivers"]:
        lines.append(f"- {driver}.")

    lines.extend(
        [
            "",
            "Important limitation: these are associative signals from a synthetic-data prototype. The output should be used to demonstrate digital-twin design, not to make clinical decisions.",
        ]
    )
    return "\n".join(lines)


def _safe_table(df: pd.DataFrame, columns: list[str], max_rows: int = 14) -> str:
    available = [c for c in columns if c in df.columns]
    if not available:
        return "No table available."
    table = df[available].copy().head(max_rows)
    for col in table.columns:
        if "date" in col:
            table[col] = pd.to_datetime(table[col]).dt.date.astype(str)
    return table.to_markdown(index=False)


def build_llm_interpretation_prompt(
    profile_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    counterfactual_df: pd.DataFrame,
    risk: dict[str, object],
    timeline_prompt: str,
    metrics: dict[str, float],
) -> str:
    """Build a compact, safety-aware prompt for hosted LLM interpretation."""

    profile = profile_df.sort_values("date").copy()
    profile["date"] = pd.to_datetime(profile["date"])
    recent = profile.tail(14)
    latest = profile.iloc[-1]
    start_date = profile["date"].min().date()
    end_date = profile["date"].max().date()

    recent_summary = {
        "participant_id": str(latest["participant_id"]),
        "history_days": int(len(profile)),
        "date_range": f"{start_date} to {end_date}",
        "recent_14d_mood_mean": round(float(recent["mood_score"].mean()), 2),
        "recent_14d_sleep_mean": round(float(recent["sleep_hours"].mean()), 2),
        "recent_14d_steps_mean": round(float(recent["steps"].mean()), 0),
        "recent_14d_stress_mean": round(float(recent["work_stress"].mean()), 2),
        "recent_14d_hrv_mean": round(float(recent["hrv_rmssd"].mean()), 2),
        "recent_14d_late_screen_mean": round(float(recent["late_screen_minutes"].mean()), 0),
    }

    forecast_table = _safe_table(forecast_df, ["date", "forecast_day", "predicted_mood", "sleep_hours", "steps", "work_stress"])
    cf_table = _safe_table(counterfactual_df, ["scenario", "mean_forecast_mood", "change_vs_baseline"], max_rows=10)
    timeline_excerpt = timeline_prompt[-7000:]

    prompt = f"""
You will write an interpretation for MoodTwin-Lite, a research/portfolio prototype.
The input combines wearable-style signals, self-reported mood, a baseline forecasting model, risk markers, counterfactual simulations, and a serialized longitudinal timeline.

Strict rules:
- Do not diagnose depression, bipolar disorder, anxiety disorder, or any medical condition.
- Do not recommend treatment, medication changes, emergency actions, or clinician instructions.
- Explain uncertainty and say this is not clinical advice.
- Treat counterfactuals as non-causal model simulations.
- Use "may be associated with" rather than causal language.
- Keep the answer concise but useful.

Required output sections:
1. Participant-friendly summary, 4-6 bullets.
2. Researcher-facing interpretation, 1 paragraph.
3. Forecast interpretation, 2-4 bullets.
4. Counterfactual interpretation, 2-4 bullets.
5. Data quality and limitations, 3-5 bullets.
6. Responsible-use note, 1 short paragraph.

Recent summary:
{recent_summary}

Model holdout metrics from the current training set:
{metrics}

Risk assessment:
{risk}

Forecast table:
{forecast_table}

Counterfactual scenario table:
{cf_table}

Serialized recent timeline excerpt:
```text
{timeline_excerpt}
```
""".strip()
    return prompt
