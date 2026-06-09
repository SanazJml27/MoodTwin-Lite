"""Generate synthetic data for MoodTwin-Lite."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.synthetic_data import SyntheticConfig, save_synthetic_data


if __name__ == "__main__":
    output = PROJECT_ROOT / "data" / "synthetic_moodtwin_profiles.csv"
    df = save_synthetic_data(
        output,
        SyntheticConfig(n_participants=12, n_days=180, start_date="2025-01-01", seed=42),
    )
    print(f"Saved {len(df):,} rows to {output}")
    print(f"Participants: {df['participant_id'].nunique()}")
