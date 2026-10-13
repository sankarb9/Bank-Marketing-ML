"""Preprocessing: feature definitions, engineered features and the scikit-learn ColumnTransformer."""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

# Raw inputs the dashboard asks for ('duration' is excluded: only known after the call -> data leakage)
RAW_FEATURES = ["age", "job", "marital", "education", "default", "balance", "housing", "loan",
                "contact", "day", "month", "campaign", "pdays", "previous", "poutcome"]
ENG_FEATURES = RAW_FEATURES + ["previously_contacted"]
NUM = ["age", "balance", "day", "campaign", "pdays", "previous"]
BIN = ["default", "housing", "loan"]
FLAG = ["previously_contacted"]
CAT = ["job", "marital", "education", "contact", "month", "poutcome"]


def add_engineered_features(df):
    """pdays = -1 means 'never contacted': turn it into a 0/1 flag and set pdays to 0."""
    out = df.copy()
    out["previously_contacted"] = (out["pdays"] != -1).astype(int)
    out["pdays"] = out["pdays"].where(out["pdays"] != -1, 0)
    return out


class QuantileClipper(BaseEstimator, TransformerMixin):
    """Winsorise columns to quantile limits learned on the training data."""
    def __init__(self, lower=0.01, upper=0.99):
        self.lower = lower; self.upper = upper
    def fit(self, X, y=None):
        X = np.asarray(X, float)
        self.lower_, self.upper_ = np.quantile(X, self.lower, axis=0), np.quantile(X, self.upper, axis=0)
        return self
    def transform(self, X):
        return np.clip(np.asarray(X, float), self.lower_, self.upper_)
    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)


def make_preprocessor():
    return ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("clip", QuantileClipper()),
                          ("scale", StandardScaler())]), NUM),
        ("bin", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                          ("enc", OrdinalEncoder(categories=[["no", "yes"]] * len(BIN),
                                                 handle_unknown="use_encoded_value", unknown_value=-1))]), BIN),
        ("flag", "passthrough", FLAG),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="constant", fill_value="unknown")),
                          ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), CAT),
    ], verbose_feature_names_out=False)
