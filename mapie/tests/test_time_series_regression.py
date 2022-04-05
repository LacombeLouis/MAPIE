from __future__ import annotations

from itertools import combinations
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import pytest
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, LeaveOneOut, train_test_split
from typing_extensions import TypedDict

from mapie._typing import NDArray
from mapie.aggregation_functions import aggregate_all
from mapie.metrics import regression_coverage_score
from mapie.time_series_regression import MapieTimeSeriesRegressor
from mapie.subsample import BlockBootstrap

X_toy = np.array(range(5)).reshape(-1, 1)
y_toy = (5.0 + 2.0 * X_toy ** 1.1).flatten()
X, y = make_regression(n_samples=500, n_features=10, noise=1.0, random_state=1)
k = np.ones(shape=(5, X.shape[1]))
METHODS = ["naive", "base", "plus", "minmax"]

Params = TypedDict(
    "Params",
    {
        "method": str,
        "agg_function": str,
        "cv": Optional[Union[int, KFold, BlockBootstrap]],
    },
)
STRATEGIES = {
    "naive": Params(method="naive", agg_function="median", cv=None),
    "jackknife": Params(method="base", agg_function="mean", cv=-1),
    "jackknife_plus": Params(method="plus", agg_function="mean", cv=-1),
    "jackknife_minmax": Params(method="minmax", agg_function="mean", cv=-1),
    "cv": Params(
        method="base",
        agg_function="mean",
        cv=KFold(n_splits=3, shuffle=True, random_state=1),
    ),
    "cv_plus": Params(
        method="plus",
        agg_function="mean",
        cv=KFold(n_splits=3, shuffle=True, random_state=1),
    ),
    "cv_minmax": Params(
        method="minmax",
        agg_function="mean",
        cv=KFold(n_splits=3, shuffle=True, random_state=1),
    ),
    "jackknife_plus_ab": Params(
        method="plus",
        agg_function="mean",
        cv=BlockBootstrap(n_resamplings=30, n_blocks=5, random_state=1),
    ),
    "jackknife_minmax_ab": Params(
        method="minmax",
        agg_function="mean",
        cv=BlockBootstrap(n_resamplings=30, n_blocks=5, random_state=1),
    ),
    "jackknife_plus_median_ab": Params(
        method="plus",
        agg_function="median",
        cv=BlockBootstrap(
            n_resamplings=30,
            n_blocks=5,
            random_state=1,
        ),
    ),
    "jackknife_plus_ab_JAB": Params(
        method="plus",
        agg_function="mean",
        cv=BlockBootstrap(n_resamplings=30, n_blocks=5, random_state=1),
    ),
    "jackknife_minmax_ab_JAB": Params(
        method="minmax",
        agg_function="mean",
        cv=BlockBootstrap(n_resamplings=30, n_blocks=5, random_state=1),
    ),
    "jackknife_plus_median_ab_JAB": Params(
        method="plus",
        agg_function="median",
        cv=BlockBootstrap(
            n_resamplings=30,
            n_blocks=5,
            random_state=1,
        ),
    ),
}

WIDTHS = {
    "naive": 3.76,
    "jackknife": 3.76,
    "jackknife_plus": 3.76,
    "jackknife_minmax": 3.82,
    "cv": 3.76,
    "cv_plus": 3.76,
    "cv_minmax": 3.95,
    "prefit": 3.89,
    "cv_plus_median": 3.90,
    "jackknife_plus_ab": 3.76,
    "jackknife_minmax_ab": 3.96,
    "jackknife_plus_median_ab": 3.76,
    "jackknife_plus_ab_JAB": 3.76,
    "jackknife_minmax_ab_JAB": 3.96,
    "jackknife_plus_median_ab_JAB": 3.76,
}

COVERAGES = {
    "naive": 0.952,
    "jackknife": 0.952,
    "jackknife_plus": 0.952,
    "jackknife_minmax": 0.952,
    "cv": 0.958,
    "cv_plus": 0.956,
    "cv_minmax": 0.956,
    "prefit": 0.90,
    "cv_plus_median": 0.954,
    "jackknife_plus_ab": 0.952,
    "jackknife_minmax_ab": 0.960,
    "jackknife_plus_median_ab": 0.946,
    "jackknife_plus_ab_JAB": 0.952,
    "jackknife_minmax_ab_JAB": 0.960,
    "jackknife_plus_median_ab_JAB": 0.965,
}


@pytest.mark.parametrize("agg_function", ["dummy", 0, 1, 2.5, [1, 2]])
def test_invalid_agg_function(agg_function: Any) -> None:
    """Test that invalid agg_functions raise errors."""

    mapie_ts_reg = MapieTimeSeriesRegressor(agg_function=None)
    with pytest.raises(ValueError, match=r".*If ensemble is True*"):
        mapie_ts_reg.fit(X_toy, y_toy)


@pytest.mark.parametrize("strategy", [*STRATEGIES])
@pytest.mark.parametrize("dataset", [(X, y), (X_toy, y_toy)])
@pytest.mark.parametrize("alpha", [0.2, [0.2, 0.4], (0.2, 0.4)])
def test_predict_output_shape(
    strategy: str, alpha: Any, dataset: Tuple[NDArray, NDArray]
) -> None:
    """Test predict output shape."""
    mapie_ts_reg = MapieTimeSeriesRegressor(**STRATEGIES[strategy])
    (X, y) = dataset
    mapie_ts_reg.fit(X, y)
    y_pred, y_pis = mapie_ts_reg.predict(X, alpha=alpha)
    n_alpha = len(alpha) if hasattr(alpha, "__len__") else 1
    assert y_pred.shape == (X.shape[0],)
    assert y_pis.shape == (X.shape[0], 2, n_alpha)


@pytest.mark.parametrize("strategy", [*STRATEGIES])
def test_results_for_same_alpha(strategy: str) -> None:
    """
    Test that predictions and intervals
    are similar with two equal values of alpha.
    """
    mapie_ts_reg = MapieTimeSeriesRegressor(**STRATEGIES[strategy])
    mapie_ts_reg.fit(X, y)
    _, y_pis = mapie_ts_reg.predict(X, alpha=[0.1, 0.1])
    np.testing.assert_allclose(y_pis[:, 0, 0], y_pis[:, 0, 1])
    np.testing.assert_allclose(y_pis[:, 1, 0], y_pis[:, 1, 1])


@pytest.mark.parametrize("strategy", [*STRATEGIES])
@pytest.mark.parametrize(
    "alpha", [np.array([0.05, 0.1]), [0.05, 0.1], (0.05, 0.1)]
)
def test_results_for_alpha_as_float_and_arraylike(
    strategy: str, alpha: Any
) -> None:
    """Test that output values do not depend on type of alpha."""
    mapie_ts_reg = MapieTimeSeriesRegressor(**STRATEGIES[strategy])
    mapie_ts_reg.fit(X, y)
    y_pred_float1, y_pis_float1 = mapie_ts_reg.predict(X, alpha=alpha[0])
    y_pred_float2, y_pis_float2 = mapie_ts_reg.predict(X, alpha=alpha[1])
    y_pred_array, y_pis_array = mapie_ts_reg.predict(X, alpha=alpha)
    np.testing.assert_allclose(y_pred_float1, y_pred_array)
    np.testing.assert_allclose(y_pred_float2, y_pred_array)
    np.testing.assert_allclose(y_pis_float1[:, :, 0], y_pis_array[:, :, 0])
    np.testing.assert_allclose(y_pis_float2[:, :, 0], y_pis_array[:, :, 1])


@pytest.mark.parametrize("strategy", [*STRATEGIES])
def test_results_for_ordered_alpha(strategy: str) -> None:
    """
    Test that prediction intervals lower (upper) bounds give
    consistent results for ordered alphas.
    """
    mapie = MapieTimeSeriesRegressor(**STRATEGIES[strategy])
    mapie.fit(X, y)
    y_pred, y_pis = mapie.predict(X, alpha=[0.05, 0.1])
    assert np.all(
        np.abs(y_pis[:, 1, 0] - y_pis[:, 0, 0])
        >= np.abs(y_pis[:, 1, 1] - y_pis[:, 0, 1])
    )


@pytest.mark.parametrize("strategy", [*STRATEGIES])
def test_results_single_and_multi_jobs(strategy: str) -> None:
    """
    Test that MapieTimeSeriesRegressor gives equal predictions
    regardless of number of parallel jobs.
    """
    mapie_single = MapieTimeSeriesRegressor(n_jobs=1, **STRATEGIES[strategy])
    mapie_multi = MapieTimeSeriesRegressor(n_jobs=-1, **STRATEGIES[strategy])
    mapie_single.fit(X_toy, y_toy)
    mapie_multi.fit(X_toy, y_toy)
    y_pred_single, y_pis_single = mapie_single.predict(X_toy, alpha=0.2)
    y_pred_multi, y_pis_multi = mapie_multi.predict(X_toy, alpha=0.2)
    np.testing.assert_allclose(y_pred_single, y_pred_multi)
    np.testing.assert_allclose(y_pis_single, y_pis_multi)

    y_pred_single_JAB, y_pis_single_JAB = mapie_single.predict(
        X_toy, alpha=0.2, JAB_Like=True
    )
    y_pred_multi_JAB, y_pis_multi_JAB = mapie_multi.predict(
        X_toy, alpha=0.2, JAB_Like=True
    )
    np.testing.assert_allclose(y_pred_single_JAB, y_pred_multi_JAB)
    np.testing.assert_allclose(y_pis_single_JAB, y_pis_multi_JAB)


@pytest.mark.parametrize("strategy", [*STRATEGIES])
def test_results_with_constant_sample_weights(strategy: str) -> None:
    """
    Test predictions when sample weights are None
    or constant with different values.
    """
    n_samples = len(X)
    mapie0 = MapieTimeSeriesRegressor(**STRATEGIES[strategy])
    mapie1 = MapieTimeSeriesRegressor(**STRATEGIES[strategy])
    mapie2 = MapieTimeSeriesRegressor(**STRATEGIES[strategy])
    mapie0.fit(X, y, sample_weight=None)
    mapie1.fit(X, y, sample_weight=np.ones(shape=n_samples))
    mapie2.fit(X, y, sample_weight=np.ones(shape=n_samples) * 5)
    y_pred0, y_pis0 = mapie0.predict(X, alpha=0.05)
    y_pred1, y_pis1 = mapie1.predict(X, alpha=0.05)
    y_pred2, y_pis2 = mapie2.predict(X, alpha=0.05)
    np.testing.assert_allclose(y_pred0, y_pred1)
    np.testing.assert_allclose(y_pred1, y_pred2)
    np.testing.assert_allclose(y_pis0, y_pis1)
    np.testing.assert_allclose(y_pis1, y_pis2)

    y_pred0_JAB, y_pis0_JAB = mapie0.predict(
        X[:200, :], alpha=0.05, JAB_Like=True
    )
    y_pred1_JAB, y_pis1_JAB = mapie1.predict(
        X[:200, :], alpha=0.05, JAB_Like=True
    )
    y_pred2_JAB, y_pis2_JAB = mapie2.predict(
        X[:200, :], alpha=0.05, JAB_Like=True
    )
    np.testing.assert_allclose(y_pred0_JAB, y_pred1_JAB)
    np.testing.assert_allclose(y_pred1_JAB, y_pred2_JAB)
    np.testing.assert_allclose(y_pis0_JAB, y_pis1_JAB)
    np.testing.assert_allclose(y_pis1_JAB, y_pis2_JAB)


@pytest.mark.parametrize("method", ["plus", "minmax"])
@pytest.mark.parametrize("cv", [-1, 2, 3, 5])
@pytest.mark.parametrize("agg_function", ["mean", "median"])
@pytest.mark.parametrize("alpha", [0.05, 0.1, 0.2])
def test_prediction_agg_function(
    method: str, cv: Union[LeaveOneOut, KFold], agg_function: str, alpha: int
) -> None:
    """
    Test that predictions differ when ensemble is True/False,
    but not prediction intervals.
    """
    mapie = MapieTimeSeriesRegressor(
        method=method, cv=cv, agg_function=agg_function
    )
    mapie.fit(X, y)
    y_pred_1, y_pis_1 = mapie.predict(X[:200, :], ensemble=True, alpha=alpha)
    y_pred_2, y_pis_2 = mapie.predict(X[:200, :], ensemble=False, alpha=alpha)
    np.testing.assert_allclose(y_pis_1[:, 0, 0], y_pis_2[:, 0, 0])
    np.testing.assert_allclose(y_pis_1[:, 1, 0], y_pis_2[:, 1, 0])
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(y_pred_1, y_pred_2)

    y_pred_1_JAB, y_pis_1_JAB = mapie.predict(
        X[:200, :], ensemble=True, alpha=alpha, JAB_Like=True
    )
    y_pred_2_JAB, y_pis_2_JAB = mapie.predict(
        X[:200, :], ensemble=False, alpha=alpha, JAB_Like=True
    )
    np.testing.assert_allclose(y_pis_1_JAB[:, 0, 0], y_pis_2_JAB[:, 0, 0])
    np.testing.assert_allclose(y_pis_1_JAB[:, 1, 0], y_pis_2_JAB[:, 1, 0])
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(y_pred_1_JAB, y_pred_2_JAB)


@pytest.mark.parametrize("strategy", [*STRATEGIES])
def test_linear_regression_results(strategy: str) -> None:
    """
    Test expected prediction intervals for
    a multivariate linear regression problem
    with fixed random state.
    """
    mapie_ts = MapieTimeSeriesRegressor(**STRATEGIES[strategy])
    mapie_ts.fit(X, y)
    if "JAB" in strategy:
        _, y_pis = mapie_ts.predict(X[:200, :], alpha=0.05, JAB_Like=True)
    else:
        _, y_pis = mapie_ts.predict(X, alpha=0.05, JAB_Like=False)
    y_pred_low, y_pred_up = y_pis[:, 0, 0], y_pis[:, 1, 0]
    width_mean = (y_pred_up - y_pred_low).mean()
    if "JAB" in strategy:
        coverage = regression_coverage_score(y[:200], y_pred_low, y_pred_up)
    else:
        coverage = regression_coverage_score(y, y_pred_low, y_pred_up)
    np.testing.assert_allclose(width_mean, WIDTHS[strategy], rtol=1e-2)
    np.testing.assert_allclose(coverage, COVERAGES[strategy], rtol=1e-2)


def test_results_prefit_ignore_method() -> None:
    """Test that method is ignored when ``cv="prefit"``."""
    estimator = LinearRegression().fit(X, y)
    all_y_pis: List[NDArray] = []
    for method in METHODS:
        mapie_ts_reg = MapieTimeSeriesRegressor(
            estimator=estimator, cv="prefit", method=method
        )
        mapie_ts_reg.fit(X, y)
        _, y_pis = mapie_ts_reg.predict(X, alpha=0.1)
        all_y_pis.append(y_pis)
    for y_pis1, y_pis2 in combinations(all_y_pis, 2):
        np.testing.assert_allclose(y_pis1, y_pis2)


def test_results_prefit_naive() -> None:
    """
    Test that prefit, fit and predict on the same dataset
    is equivalent to the "naive" method.
    """
    estimator = LinearRegression().fit(X, y)
    mapie_ts_reg = MapieTimeSeriesRegressor(estimator=estimator, cv="prefit")
    mapie_ts_reg.fit(X, y)
    _, y_pis = mapie_ts_reg.predict(X, alpha=0.05)
    width_mean = (y_pis[:, 1, 0] - y_pis[:, 0, 0]).mean()
    coverage = regression_coverage_score(y, y_pis[:, 0, 0], y_pis[:, 1, 0])
    np.testing.assert_allclose(width_mean, WIDTHS["naive"], rtol=1e-2)
    np.testing.assert_allclose(coverage, COVERAGES["naive"], rtol=1e-2)


def test_results_prefit() -> None:
    """Test prefit results on a standard train/validation/test split."""
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=1 / 10, random_state=1
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=1 / 9, random_state=1
    )
    estimator = LinearRegression().fit(X_train, y_train)
    mapie_ts_reg = MapieTimeSeriesRegressor(estimator=estimator, cv="prefit")
    mapie_ts_reg.fit(X_val, y_val)
    _, y_pis = mapie_ts_reg.predict(X_test, alpha=0.05)
    width_mean = (y_pis[:, 1, 0] - y_pis[:, 0, 0]).mean()
    coverage = regression_coverage_score(
        y_test, y_pis[:, 0, 0], y_pis[:, 1, 0]
    )
    np.testing.assert_allclose(width_mean, WIDTHS["prefit"], rtol=1e-2)
    np.testing.assert_allclose(coverage, COVERAGES["prefit"], rtol=1e-2)


def test_not_enough_resamplings() -> None:
    """Test that a warning is raised if at least one residual is nan."""
    with pytest.warns(UserWarning, match=r"WARNING: at least one point of*"):
        mapie_ts_reg = MapieTimeSeriesRegressor(
            cv=BlockBootstrap(n_resamplings=1, n_blocks=1), agg_function="mean"
        )
        mapie_ts_reg.fit(X, y)


def test_no_agg_fx_specified_with_subsample() -> None:
    """Test that a warning is raised if at least one residual is nan."""
    with pytest.raises(
        ValueError, match=r"You need to specify an aggregation*"
    ):
        mapie_ts_reg = MapieTimeSeriesRegressor(
            cv=BlockBootstrap(n_resamplings=1, n_blocks=1),
            agg_function=None,
        )
        mapie_ts_reg.fit(X, y)


def test_invalid_aggregate_all() -> None:
    """
    Test that wrong aggregation in MAPIE raise errors.
    """
    with pytest.raises(
        ValueError,
        match=r".*Aggregation function called but not defined.*",
    ):
        aggregate_all(None, X)


def test_aggregate_with_mask_with_prefit() -> None:
    """
    Test ``aggregate_with_mask`` in case ``cv`` is ``"prefit"``.
    """
    mapie_ts_reg = MapieTimeSeriesRegressor(cv="prefit")
    with pytest.raises(
        ValueError,
        match=r".*There should not be aggregation of predictions if cv is*",
    ):
        mapie_ts_reg.aggregate_with_mask(k, k)

    mapie_ts_reg = MapieTimeSeriesRegressor(agg_function="nonsense")
    with pytest.raises(
        ValueError,
        match=r".*The value of self.agg_function is not correct*",
    ):
        mapie_ts_reg.aggregate_with_mask(k, k)


def test_pred_loof_isnan() -> None:
    """Test that if validation set is empty then prediction is empty."""
    mapie_ts_reg = MapieTimeSeriesRegressor()
    _, y_pred, _ = mapie_ts_reg._fit_and_predict_oof_model(
        estimator=mapie_ts_reg,
        X=X_toy,
        y=y_toy,
        train_index=[0, 1, 2, 3, 4],
        val_index=[],
    )
    assert len(y_pred) == 0


def test_MapieTimeSeriesRegressor_alpha_is_None() -> None:
    """Test ``predict`` when ``alpha`` is None."""
    mapie_ts_reg = MapieTimeSeriesRegressor(cv=-1).fit(X_toy, y_toy)

    with pytest.raises(ValueError, match=r".*too many values to unpackt*"):
        y_pred, y_pis = mapie_ts_reg.predict(X_toy, alpha=None)


def test_MapieTimeSeriesRegressor_partial_fit_ensemble() -> None:
    """Test ``partial_fit``."""
    mapie_ts_reg = MapieTimeSeriesRegressor(cv=-1).fit(X_toy, y_toy)
    assert round(mapie_ts_reg.conformity_scores_[-1], 2) == round(
        np.abs(14.189 - 14.038), 2
    )
    mapie_ts_reg = mapie_ts_reg.partial_fit(
        X=np.array([[6]]), y=np.array([17.5])
    )
    assert round(mapie_ts_reg.conformity_scores_[-1], 2) == round(
        17.5 - 18.665, 2
    )
