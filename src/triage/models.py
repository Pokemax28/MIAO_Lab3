"""Model definitions, one per released version."""

from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler


def build_v01() -> Pipeline:
    """Baseline: standardize the features, then ordinary least squares."""
    return make_pipeline(StandardScaler(), LinearRegression())


MODEL_BUILDERS = {
    "v0.1": build_v01,
}


def build_model(version: str):
    """Return an unfitted model for the requested version."""
    try:
        return MODEL_BUILDERS[version]()
    except KeyError:
        known = ", ".join(sorted(MODEL_BUILDERS))
        raise ValueError(f"Unknown model version '{version}'. Known versions: {known}") from None
