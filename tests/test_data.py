import numpy as np
import pandas as pd
import pytest

from tftk.data import TsData

orig_data_dim = (100, 5)
data = np.random.randint(0, 10, orig_data_dim)
date_rng = pd.date_range("20180101 00:00", "20180105 03:00", freq="1H")
cols = ["a", "b", "c", "d", "e"]
predictor_labels = ["a", "b", "c"]
target_label = ["d"]
df = pd.DataFrame(data=data, columns=cols, index=date_rng)


@pytest.fixture
def sample_ts_data():
    return TsData.make_ts_features(
        ts_df=df,
        predictor_labels=predictor_labels,
        target_label=target_label,
        goback_steps=5,
        period_per_step=1,
        forecast_period=3,
        auto_feature_select=False,
    )


@pytest.mark.parametrize(
    "goback_steps, period_per_step, forecast_period", [(5, 1, 3), (5, 2, 3), (7, 2, 4)]
)
def test_tsdata_make_ts_features_dims(goback_steps, period_per_step, forecast_period):

    ts_data = TsData.make_ts_features(
        ts_df=df,
        predictor_labels=predictor_labels,
        target_label=target_label,
        goback_steps=goback_steps,
        period_per_step=period_per_step,
        forecast_period=forecast_period,
        auto_feature_select=False,
    )

    pred_size = len(predictor_labels)
    feature_size = pred_size * goback_steps
    tsdata_size = (
        orig_data_dim[0] - (goback_steps - 1) * period_per_step - forecast_period
    )

    assert ts_data.ts_features.shape == (tsdata_size, feature_size)
    assert ts_data.ts_target.shape == (tsdata_size, len(target_label))


# Check if normalize will alter the dimensions of dataset
def test_normalize(sample_ts_data):
    before_norm_dim = sample_ts_data.ts_features.shape
    sample_ts_data.normalize()
    after_norm_dim = sample_ts_data.ts_features.shape
    assert before_norm_dim == after_norm_dim


# Check if smooth function run without error
def test_smooth(sample_ts_data):
    sample_ts_data.smooth(
        ma_group=["a", "b", "c"], ma_window=[3, 3, 3], exponential=True
    )


# Check if extract_stat_features function run without error
def test_extrat_stat_features(sample_ts_data):
    sample_ts_data.extract_stat_features(
        predictors=["a", "b", "c"], mean=True, median=True, volatility=True
    )
