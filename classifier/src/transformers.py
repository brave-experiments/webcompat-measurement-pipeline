import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class DropCorrelatedFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, threshold):
        self.threshold = threshold

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X)
        X_df_corr = X_df.corr(method='pearson', min_periods=1)
        non_correlated = ~(X_df_corr.mask(np.tril(np.ones([len(X_df_corr)]*2, dtype=bool))).abs() > self.threshold).any()
        self.non_correlated_features = non_correlated[non_correlated == True].index

        return self

    def transform(self, X, y=None, **kwargs):
        X_df = pd.DataFrame(X)
        return X_df[self.non_correlated_features]


class DropMajorityNullFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, threshold):
        self.threshold = threshold

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X)
        nas_df = X_df.isna().mean()
        self.feature_mask = nas_df[nas_df < self.threshold].index

        return self

    def transform(self, X, y=None, **kwargs):
        X_df = pd.DataFrame(X)
        return X_df[self.feature_mask]
