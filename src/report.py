"""Markdown report generation."""

from __future__ import annotations

from datetime import datetime

import pandas as pd


def _markdown_table(df: pd.DataFrame) -> str:
    return df.to_markdown(index=False)


def build_markdown_report(
    profile_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    counterfactual_df: pd.DataFrame,
    risk: dict[str, object],
    explanation: str,
    timeline_prompt: str,
    metrics: dict[str, float],
    llm_interpretation: str | None = None,
) -> str:
    """Build an exportable Markdown report."""

    profile = profile_df.sort_values("date")
    first = profile.iloc[0]
    latest = profile.iloc[-1]

    forecast_table = forecast_df[["date", "forecast_day", "predicted_mood"]].copy()
    forecast_table["date"] = pd.to_datetime(forecast_table["date"]).dt.date.astype(str)

    lines = [
        "# MoodTwin-Lite Digital Twin Report",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "## Disclaimer",
        "This report is produced by an educational/research prototype. It is not a medical device and must not be used for diagnosis, treatment, or clinical decision-making.",
        "",
        "## Participant Summary",
        f"- Participant ID: `{first['participant_id']}`",
        f"- Age: {int(first['age'])}",
        f"- Sex: {first['sex']}",
        f"- Baseline risk group: {first['baseline_risk_group']}",
        f"- History window: {profile['date'].min().date()} to {profile['date'].max().date()} ({len(profile)} days)",
        "",
        "## Latest Day",
        f"- Date: {latest['date'].date()}",
        f"- Mood: {latest['mood_score']}/10",
        f"- Sleep: {latest['sleep_hours']} hours",
        f"- Steps: {int(latest['steps'])}",
        f"- Stress: {int(latest['work_stress'])}/10",
        "",
        "## Model Performance on Synthetic Holdout Data",
        f"- MAE: {metrics.get('mae', float('nan')):.3f}",
        f"- RMSE: {metrics.get('rmse', float('nan')):.3f}",
        f"- R²: {metrics.get('r2', float('nan')):.3f}",
        "",
        "## Forecast",
        _markdown_table(forecast_table),
        "",
        "## Risk Assessment",
        f"- Risk level: **{risk['risk_level']}**",
        f"- Recent 7-day mean mood: {risk['recent_7d_mood_mean']}/10",
        f"- Forecast 7-day mean mood: {risk['forecast_7d_mood_mean']}/10",
        f"- Forecast change: {risk['forecast_change']}",
        "",
        "### Risk markers",
    ]

    for driver in risk["drivers"]:
        lines.append(f"- {driver}")

    lines.extend(
        [
            "",
            "## Counterfactual Scenarios",
            _markdown_table(counterfactual_df),
            "",
            "## Deterministic Explanation",
            explanation,
        ]
    )

    if llm_interpretation:
        lines.extend(
            [
                "",
                "## Optional LLM Interpretation",
                llm_interpretation,
            ]
        )

    lines.extend(
        [
            "",
            "## LLM-Ready Serialized Timeline",
            "```text",
            timeline_prompt,
            "```",
        ]
    )

    return "\n".join(lines)
