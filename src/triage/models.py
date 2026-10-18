import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, VotingRegressor
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from triage.config import SEED

RISK_FLAG_VERSIONS = {"v0.2"}


def build_v01() -> Pipeline:
    return make_pipeline(StandardScaler(), LinearRegression())


def build_ridge() -> Pipeline:
    return make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 4, 40)))


def build_gbr() -> GradientBoostingRegressor:
    return GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=2,
        subsample=0.7,
        min_samples_leaf=10,
        random_state=SEED,
    )


def build_v02() -> VotingRegressor:
    return VotingRegressor([("ridge", build_ridge()), ("gbr", build_gbr())])


MODEL_BUILDERS = {
    "v0.1": build_v01,
    "v0.2": build_v02,
}


def build_model(version: str):
    try:
        return MODEL_BUILDERS[version]()
    except KeyError:
        known = ", ".join(sorted(MODEL_BUILDERS))
        raise ValueError(f"Unknown model version '{version}'. Known versions: {known}") from None
