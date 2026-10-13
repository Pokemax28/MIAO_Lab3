"""Model definitions, one per released version."""

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, VotingRegressor
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from triage.config import SEED

# Versions whose training also calibrates a high-risk threshold.
RISK_FLAG_VERSIONS = {"v0.2"}


def build_v01() -> Pipeline:
    """Baseline: standardize the features, then ordinary least squares."""
    return make_pipeline(StandardScaler(), LinearRegression())


def build_ridge() -> Pipeline:
    """Ridge regression whose penalty is picked by built-in leave-one-out validation."""
    return make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 4, 40)))


def build_gbr() -> GradientBoostingRegressor:
    """Small, heavily regularized gradient boosting (shallow trees, subsampling)."""
    return GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=2,
        subsample=0.7,
        min_samples_leaf=10,
        random_state=SEED,
    )


def build_v02() -> VotingRegressor:
    """Average of a regularized linear model and a boosted tree model."""
    return VotingRegressor([("ridge", build_ridge()), ("gbr", build_gbr())])


MODEL_BUILDERS = {
    "v0.1": build_v01,
    "v0.2": build_v02,
}


def build_model(version: str):
    """Return an unfitted model for the requested version."""
    try:
        return MODEL_BUILDERS[version]()
    except KeyError:
        known = ", ".join(sorted(MODEL_BUILDERS))
        raise ValueError(f"Unknown model version '{version}'. Known versions: {known}") from None
