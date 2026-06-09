from src.serializer import serialize_profile_timeline
from src.synthetic_data import SyntheticConfig, generate_synthetic_data


def test_serializer_contains_participant_and_timeline():
    df = generate_synthetic_data(SyntheticConfig(n_participants=1, n_days=20, seed=1))
    text = serialize_profile_timeline(df, last_n_days=10)
    assert "Participant summary" in text
    assert "Daily timeline" in text
    assert "forecast mood_score" in text
    assert "MT001" in text
