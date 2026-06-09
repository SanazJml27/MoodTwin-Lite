from pathlib import Path

import pandas as pd

from src.public_datasets import adapt_ifh_affect_dataset
from src.schemas import REQUIRED_COLUMNS


def test_adapt_ifh_affect_minimal_fixture(tmp_path: Path):
    par = tmp_path / "par_1"
    (par / "ema").mkdir(parents=True)
    (par / "oura").mkdir(parents=True)

    pd.DataFrame(
        {
            "submission_timestamp": [1767225600000, 1767312000000],
            "Active": [80, 40],
            "Alert": [70, 45],
            "Distressed": [10, 50],
            "Nervous": [20, 60],
        }
    ).to_csv(par / "ema" / "daily.csv", index=False)
    pd.DataFrame({"summary_date": ["2026-01-01", "2026-01-02"], "steps": [9000, 5000]}).to_csv(
        par / "oura" / "activity.csv", index=False
    )
    pd.DataFrame(
        {
            "summary_date": ["2026-01-01", "2026-01-02"],
            "total_sleep_duration": [27000, 21600],
            "efficiency": [90, 78],
        }
    ).to_csv(par / "oura" / "sleep.csv", index=False)
    pd.DataFrame(
        {
            "summary_date": ["2026-01-01", "2026-01-02"],
            "average_hrv": [55, 38],
            "lowest_resting_heart_rate": [61, 69],
        }
    ).to_csv(par / "oura" / "readiness.csv", index=False)

    out = adapt_ifh_affect_dataset(tmp_path)
    assert list(out.columns) == REQUIRED_COLUMNS
    assert len(out) == 2
    assert out["participant_id"].iloc[0] == "IFH1"
    assert out["mood_score"].iloc[0] > out["mood_score"].iloc[1]
