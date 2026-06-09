"""Data schemas for MoodTwin-Lite."""

from __future__ import annotations

from datetime import date
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator


class WearableDay(BaseModel):
    """One day of mood + wearable/smart-device data."""

    participant_id: str
    date: date
    age: int = Field(ge=13, le=100)
    sex: Literal["female", "male", "other"]
    baseline_risk_group: Literal["low", "moderate", "elevated"]

    sleep_hours: float = Field(ge=0, le=16)
    sleep_efficiency: float = Field(ge=0, le=100)
    sleep_midpoint_hour: float = Field(ge=0, le=24)

    steps: int = Field(ge=0, le=60000)
    active_minutes: int = Field(ge=0, le=600)
    resting_hr: float = Field(ge=30, le=140)
    hrv_rmssd: float = Field(ge=1, le=250)

    screen_minutes: int = Field(ge=0, le=1440)
    late_screen_minutes: int = Field(ge=0, le=600)
    social_minutes: int = Field(ge=0, le=600)
    work_stress: int = Field(ge=1, le=10)
    medication_adherence: int = Field(ge=0, le=1)

    mood_score: float = Field(ge=1, le=10)
    anxiety_score: float = Field(ge=1, le=10)
    energy_score: float = Field(ge=1, le=10)
    notes: Optional[str] = ""

    @field_validator("participant_id")
    @classmethod
    def participant_id_not_empty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("participant_id cannot be empty")
        return value


REQUIRED_COLUMNS = list(WearableDay.model_fields.keys())
