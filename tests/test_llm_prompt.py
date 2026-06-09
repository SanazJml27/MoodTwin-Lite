import pandas as pd

from src.counterfactuals import run_counterfactuals
from src.llm_explainer import build_llm_interpretation_prompt
from src.llm_provider import default_model_for_provider, normalize_provider
from src.model import forecast_mood, train_next_day_mood_model
from src.risk import assess_deterioration_risk
from src.serializer import serialize_profile_timeline
from src.synthetic_data import SyntheticConfig, generate_synthetic_data


def test_llm_prompt_contains_safety_and_required_sections():
    df = generate_synthetic_data(SyntheticConfig(n_participants=3, n_days=80, seed=123))
    model_result = train_next_day_mood_model(df)
    profile = df[df["participant_id"] == df["participant_id"].iloc[0]].sort_values("date")
    forecast = forecast_mood(profile, model_result.model, days=7)
    risk = assess_deterioration_risk(profile, forecast)
    counterfactuals = run_counterfactuals(profile, model_result.model, days=7)
    timeline = serialize_profile_timeline(profile, last_n_days=14)

    prompt = build_llm_interpretation_prompt(
        profile_df=profile,
        forecast_df=forecast,
        counterfactual_df=counterfactuals,
        risk=risk,
        timeline_prompt=timeline,
        metrics=model_result.metrics,
    )

    assert "Do not diagnose" in prompt
    assert "Counterfactual" in prompt
    assert "Serialized recent timeline" in prompt
    assert len(prompt) > 1000


def test_provider_defaults_are_safe():
    assert normalize_provider("unknown") == "none"
    assert default_model_for_provider("openai")
    assert default_model_for_provider("gemini")
