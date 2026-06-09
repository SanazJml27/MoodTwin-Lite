"""Forecasting model and recursive prediction helpers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.features import PREDICTOR_COLUMNS, add_time_features, clean_input_dataframe, prepare_supervised_data


@dataclass
class ModelResult:
    model: Pipeline | RandomForestRegressor
    metrics: dict[str, float]
    supervised_rows: int


def train_next_day_mood_model(df: pd.DataFrame, model_type: str = "gradient_boosting") -> ModelResult:
    """Train a small baseline model for next-day mood prediction."""

    X, y, supervised = prepare_supervised_data(df)
    if len(X) < 50:
        raise ValueError("Need at least 50 supervised rows to train the model.")

    # Random split is acceptable for a starter demo. A stronger version should use temporal validation.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    if model_type == "random_forest":
        model = RandomForestRegressor(n_estimators=80, min_samples_leaf=3, random_state=42, n_jobs=1)
    else:
        model = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "regressor",
                    GradientBoostingRegressor(
                        n_estimators=120,
                        learning_rate=0.05,
                        max_depth=3,
                        random_state=42,
                    ),
                ),
            ]
        )

    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    metrics = {
        "mae": float(mean_absolute_error(y_test, preds)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, preds))),
        "r2": float(r2_score(y_test, preds)),
    }
    return ModelResult(model=model, metrics=metrics, supervised_rows=len(supervised))


def _clip(value: float, lower: float, upper: float) -> float:
    return float(max(lower, min(upper, value)))


def _future_baseline_values(profile_df: pd.DataFrame, scenario: Optional[dict[str, float]] = None) -> dict[str, float]:
    """Estimate future daily covariates from the participant's recent 7-day averages."""

    recent = clean_input_dataframe(profile_df).tail(7)
    values = {
        "sleep_hours": float(recent["sleep_hours"].mean()),
        "sleep_efficiency": float(recent["sleep_efficiency"].mean()),
        "sleep_midpoint_hour": float(recent["sleep_midpoint_hour"].mean()),
        "steps": float(recent["steps"].mean()),
        "active_minutes": float(recent["active_minutes"].mean()),
        "resting_hr": float(recent["resting_hr"].mean()),
        "hrv_rmssd": float(recent["hrv_rmssd"].mean()),
        "screen_minutes": float(recent["screen_minutes"].mean()),
        "late_screen_minutes": float(recent["late_screen_minutes"].mean()),
        "social_minutes": float(recent["social_minutes"].mean()),
        "work_stress": float(recent["work_stress"].mean()),
        "medication_adherence": float(recent["medication_adherence"].mean()),
    }

    scenario = scenario or {}
    for key, delta in scenario.items():
        if key in values:
            values[key] += delta

    values["sleep_hours"] = _clip(values["sleep_hours"], 3.0, 10.5)
    values["sleep_efficiency"] = _clip(values["sleep_efficiency"], 55, 98)
    values["steps"] = _clip(values["steps"], 500, 25000)
    values["active_minutes"] = _clip(values["active_minutes"], 0, 240)
    values["resting_hr"] = _clip(values["resting_hr"], 45, 115)
    values["hrv_rmssd"] = _clip(values["hrv_rmssd"], 8, 140)
    values["screen_minutes"] = _clip(values["screen_minutes"], 30, 900)
    values["late_screen_minutes"] = _clip(values["late_screen_minutes"], 0, 360)
    values["social_minutes"] = _clip(values["social_minutes"], 0, 360)
    values["work_stress"] = _clip(values["work_stress"], 1, 10)
    values["medication_adherence"] = _clip(values["medication_adherence"], 0, 1)
    return values


def forecast_mood(
    profile_df: pd.DataFrame,
    model,
    days: int = 7,
    scenario: Optional[dict[str, float]] = None,
) -> pd.DataFrame:
    """Forecast mood recursively for a participant.

    The unknown future wearable values are approximated using recent averages plus
    optional scenario deltas. The predicted mood is then fed back as the lagged
    mood for the following day.
    """

    history = clean_input_dataframe(profile_df).copy()
    history = history.sort_values("date").reset_index(drop=True)
    if history.empty:
        raise ValueError("profile_df is empty")

    future_values = _future_baseline_values(history, scenario)
    forecasts = []
    mood_history = history["mood_score"].astype(float).tolist()

    last_date = history["date"].max()
    participant_id = history["participant_id"].iloc[0]

    for i in range(1, days + 1):
        forecast_date = last_date + timedelta(days=i)
        dow = forecast_date.dayofweek
        row = {
            "participant_id": participant_id,
            "date": forecast_date,
            **future_values,
            "day_of_week_sin": np.sin(2 * np.pi * dow / 7),
            "day_of_week_cos": np.cos(2 * np.pi * dow / 7),
            "mood_lag1": float(mood_history[-1]),
            "mood_rolling3": float(np.mean(mood_history[-3:])),
            "mood_rolling7": float(np.mean(mood_history[-7:])),
            "sleep_rolling7": float(history["sleep_hours"].tail(7).mean()),
            "steps_rolling7": float(history["steps"].tail(7).mean()),
            "hrv_rolling7": float(history["hrv_rmssd"].tail(7).mean()),
            "stress_rolling7": float(history["work_stress"].tail(7).mean()),
        }
        X_future = pd.DataFrame([row])[PREDICTOR_COLUMNS]
        predicted = float(model.predict(X_future)[0])
        predicted = _clip(predicted, 1, 10)
        row["predicted_mood"] = round(predicted, 2)
        row["forecast_day"] = i
        forecasts.append(row)
        mood_history.append(predicted)

    return pd.DataFrame(forecasts)


def feature_importance_table(model_result: ModelResult) -> pd.DataFrame:
    """Return feature importances when available.

    GradientBoostingRegressor inside a pipeline does not expose simple feature importances here, so
    this returns an empty dataframe for that model. Random forest users will get
    native importances.
    """

    model = model_result.model
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        return pd.DataFrame({"feature": PREDICTOR_COLUMNS, "importance": importances}).sort_values(
            "importance", ascending=False
        )
    return pd.DataFrame(columns=["feature", "importance"])
