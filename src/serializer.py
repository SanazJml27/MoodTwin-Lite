"""Timeline serialization inspired by patient digital-twin frameworks.

This module converts daily structured data into compact, human-readable text.
The output can be used as an LLM prompt or as a transparent audit trail for the
forecasting app.
"""

from __future__ import annotations

import pandas as pd


def serialize_profile_timeline(profile_df: pd.DataFrame, last_n_days: int = 30) -> str:
    """Serialize recent participant history into an LLM-ready prompt."""

    if profile_df.empty:
        raise ValueError("profile_df cannot be empty")

    df = profile_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    recent = df.tail(last_n_days)
    first = df.iloc[0]
    last = df.iloc[-1]

    lines = [
        "You are given a longitudinal digital-phenotyping timeline for one participant.",
        "The task is to forecast near-term mood trajectory and identify non-causal risk markers.",
        "Do not provide diagnosis or treatment advice.",
        "",
        "Participant summary:",
        f"- participant_id: {first['participant_id']}",
        f"- age: {int(first['age'])}",
        f"- sex: {first['sex']}",
        f"- baseline_risk_group: {first['baseline_risk_group']}",
        f"- available_history: {df['date'].min().date()} to {df['date'].max().date()} ({len(df)} days)",
        "",
        "Recent aggregate state:",
        f"- last_7d_mean_mood: {df['mood_score'].tail(7).mean():.2f}/10",
        f"- last_7d_mean_sleep_hours: {df['sleep_hours'].tail(7).mean():.2f}",
        f"- last_7d_mean_steps: {df['steps'].tail(7).mean():.0f}",
        f"- last_7d_mean_hrv_rmssd: {df['hrv_rmssd'].tail(7).mean():.1f}",
        f"- last_7d_mean_stress: {df['work_stress'].tail(7).mean():.2f}/10",
        f"- last_7d_late_screen_minutes: {df['late_screen_minutes'].tail(7).mean():.0f}",
        "",
        f"Daily timeline, last {len(recent)} days:",
    ]

    for _, row in recent.iterrows():
        lines.append(
            "- "
            f"{row['date'].date()}: "
            f"mood={row['mood_score']:.1f}/10, "
            f"anxiety={row['anxiety_score']:.1f}/10, "
            f"energy={row['energy_score']:.1f}/10, "
            f"sleep={row['sleep_hours']:.1f}h, "
            f"sleep_eff={row['sleep_efficiency']:.0f}%, "
            f"steps={int(row['steps'])}, "
            f"active_min={int(row['active_minutes'])}, "
            f"rest_hr={row['resting_hr']:.0f}, "
            f"hrv={row['hrv_rmssd']:.0f}, "
            f"screen={int(row['screen_minutes'])}min, "
            f"late_screen={int(row['late_screen_minutes'])}min, "
            f"social={int(row['social_minutes'])}min, "
            f"stress={int(row['work_stress'])}/10, "
            f"med_adherence={int(row['medication_adherence'])}, "
            f"note='{row.get('notes', '')}'"
        )

    lines.extend(
        [
            "",
            "Forecasting question:",
            "- Based on the recent trajectory, forecast mood_score for the next 7 days.",
            "- Explain which recent non-causal markers appear most associated with the forecast.",
            "- State uncertainty and limitations clearly.",
            "- Avoid diagnosis, medication advice, or emergency guidance.",
        ]
    )
    return "\n".join(lines)
