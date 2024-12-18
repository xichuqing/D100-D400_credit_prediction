import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class LogCap(BaseEstimator, TransformerMixin):
    def __init__(self, lower_quantile=0.05, upper_quantile=0.95):
        """
        Applies quantile capping (winsorization) and log transformation.

        Parameters:
        - lower_quantile: Lower quantile threshold for capping.
        - upper_quantile: Upper quantile threshold for capping.
        """
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile

    def fit(self, X, y=None):
        """
        Fit the transformer to compute the quantile thresholds.

        Parameters:
        - X: Input array.
        - y: Ignored (for compatibility).

        Returns:
        - self: Fitted transformer.
        """
        self.lower_bound_ = np.percentile(X, self.lower_quantile * 100)
        self.upper_bound_ = np.percentile(X, self.upper_quantile * 100)
        return self

    def transform(self, X):
        """
        Apply quantile capping and log transformation.

        Parameters:
        - X: Input array.

        Returns:
        - X_transformed: Array after capping and log transformation.
        """
        check_is_fitted(self)
        X_clipped = np.clip(X, self.lower_bound_, self.upper_bound_)
        # Apply log transformation (add a small constant to avoid log(0))
        X_transformed = np.log1p(X_clipped)
        return X_transformed
