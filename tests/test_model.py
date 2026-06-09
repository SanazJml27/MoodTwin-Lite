from src.model import forecast_mood, train_next_day_mood_model
from src.synthetic_data import SyntheticConfig, generate_synthetic_data


def test_train_and_forecast():
    df = generate_synthetic_data(SyntheticConfig(n_participants=4, n_days=70, seed=3))
    result = train_next_day_mood_model(df)
    profile = df[df["participant_id"] == "MT001"]
    forecast = forecast_mood(profile, result.model, days=7)
    assert len(forecast) == 7
    assert forecast["predicted_mood"].between(1, 10).all()
