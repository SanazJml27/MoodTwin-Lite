from src.features import PREDICTOR_COLUMNS, prepare_supervised_data
from src.synthetic_data import SyntheticConfig, generate_synthetic_data


def test_prepare_supervised_data_has_predictors():
    df = generate_synthetic_data(SyntheticConfig(n_participants=2, n_days=40, seed=2))
    X, y, supervised = prepare_supervised_data(df)
    assert not X.empty
    assert not y.empty
    assert set(PREDICTOR_COLUMNS).issubset(X.columns)
    assert len(X) == len(y) == len(supervised)
