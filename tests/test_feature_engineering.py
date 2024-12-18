import numpy as np
import pytest
import sys
import os
from pathlib import Path
root =Path(__file__).resolve().parent.parent
sys.path.append(str(root))
from preprocessing.feature_engineering import LogCap


@pytest.mark.parametrize(
    "lower_quantile, upper_quantile", [(0.05, 0.95), (0.1, 0.9), (0.2, 0.8)]
)
def test_log_winsorizer(lower_quantile, upper_quantile):
    """
    Test the LogCap transformer for quantile capping and log transformation.
    """
    # Generate random data
    X = np.random.exponential(scale=2, size=1000)

    # Fit and transform using LogWinsorizer
    transformer = LogCap(
        lower_quantile=lower_quantile, upper_quantile=upper_quantile
    )
    Xt = transformer.fit_transform(X)

    # Calculate expected quantile bounds
    lower_bound = np.percentile(X, lower_quantile * 100)
    upper_bound = np.percentile(X, upper_quantile * 100)

    # Assert that the max and min values of transformed data match the quantiles
    assert np.allclose(Xt.max(), np.log1p(upper_bound)), "Upper bound mismatch"
    assert np.allclose(Xt.min(), np.log1p(lower_bound)), "Lower bound mismatch"

    # Assert no NaN values in transformed data
    assert not np.isnan(Xt).any(), "Transformed data contains NaN"
