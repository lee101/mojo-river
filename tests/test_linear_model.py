import inspect

import numpy as np
import pandas as pd
import pytest

from river import linear_model as river_linear
from river import optim as river_optim

from mojoriver import linear_model, optim


@pytest.fixture
def stream():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(80, 7))
    y = X @ np.array([1.2, -0.5, 0.0, 2.0, -1.0, 0.3, 0.8]) + 0.2
    return X, y


@pytest.mark.parametrize(
    ("ours_optimizer", "river_optimizer"),
    [
        (optim.SGD(0.015), river_optim.SGD(0.015)),
        (optim.AdaGrad(0.05), river_optim.AdaGrad(0.05)),
        (optim.RMSProp(0.02, rho=0.85), river_optim.RMSProp(0.02, rho=0.85)),
        (optim.Adam(0.02), river_optim.Adam(0.02)),
    ],
)
def test_linear_regression_learn_one_parity(stream, ours_optimizer, river_optimizer):
    X, y = stream
    ours = linear_model.LinearRegression(
        optimizer=ours_optimizer, l2=0.03, intercept_lr=0.02, clip_gradient=10
    )
    theirs = river_linear.LinearRegression(
        optimizer=river_optimizer, l2=0.03, intercept_lr=0.02, clip_gradient=10
    )
    for row, target in zip(X, y):
        x = {f"f{i}": value for i, value in enumerate(row)}
        assert ours.learn_one(x, target) is None
        theirs.learn_one(x, target)
    assert ours.weights == pytest.approx(theirs.weights, abs=2e-14)
    assert ours.intercept == pytest.approx(theirs.intercept, abs=2e-14)
    probe = {f"f{i}": value for i, value in enumerate(X[0])}
    assert ours.predict_one(probe) == pytest.approx(theirs.predict_one(probe), abs=2e-14)


@pytest.mark.parametrize(
    ("ours_optimizer", "river_optimizer"),
    [
        (optim.SGD(0.03), river_optim.SGD(0.03)),
        (optim.AdaGrad(0.08), river_optim.AdaGrad(0.08)),
        (optim.RMSProp(0.03), river_optim.RMSProp(0.03)),
        (optim.Adam(0.03), river_optim.Adam(0.03)),
    ],
)
def test_logistic_regression_learn_one_parity(stream, ours_optimizer, river_optimizer):
    X, target = stream
    y = target > np.median(target)
    ours = linear_model.LogisticRegression(optimizer=ours_optimizer, l2=0.02)
    theirs = river_linear.LogisticRegression(optimizer=river_optimizer, l2=0.02)
    for row, label in zip(X, y):
        x = dict(enumerate(row))
        ours.learn_one(x, label)
        theirs.learn_one(x, label)
    assert ours.weights == pytest.approx(theirs.weights, abs=2e-13)
    assert ours.intercept == pytest.approx(theirs.intercept, abs=2e-14)
    for row in X[:10]:
        x = dict(enumerate(row))
        assert ours.predict_proba_one(x) == pytest.approx(
            theirs.predict_proba_one(x), abs=2e-13
        )
        assert ours.predict_one(x) == theirs.predict_one(x)


@pytest.mark.parametrize("logistic", [False, True])
def test_learn_many_and_predict_many_parity(stream, logistic):
    X_values, target = stream
    X = pd.DataFrame(X_values, columns=list("abcdefg"))
    y = pd.Series(target > 0, name="target") if logistic else pd.Series(target, name="target")
    if logistic:
        ours = linear_model.LogisticRegression(optim.Adam(0.04), l2=0.1)
        theirs = river_linear.LogisticRegression(river_optim.Adam(0.04), l2=0.1)
    else:
        ours = linear_model.LinearRegression(optim.Adam(0.04), l2=0.1)
        theirs = river_linear.LinearRegression(river_optim.Adam(0.04), l2=0.1)
    weights = pd.Series(np.linspace(0.5, 1.5, len(X)))
    for indices in np.array_split(np.arange(len(X)), 4):
        ours.learn_many(X.iloc[indices], y.iloc[indices], weights.iloc[indices])
        theirs.learn_many(X.iloc[indices], y.iloc[indices], weights.iloc[indices])
    assert ours.weights == pytest.approx(theirs.weights, abs=2e-13)
    assert ours.intercept == pytest.approx(theirs.intercept, abs=2e-14)
    if logistic:
        assert np.allclose(ours.predict_proba_many(X), theirs.predict_proba_many(X))
        assert np.array_equal(ours.predict_many(X), theirs.predict_many(X))
    else:
        assert np.allclose(ours.predict_many(X), theirs.predict_many(X))
        assert ours.predict_many(X).name == theirs.predict_many(X).name


@pytest.mark.parametrize("mode", [0, 1, 2])
def test_pa_regressor_parity(stream, mode):
    X, y = stream
    ours = linear_model.PARegressor(C=0.07, mode=mode, eps=0.05)
    theirs = river_linear.PARegressor(C=0.07, mode=mode, eps=0.05)
    for row, target in zip(X, y):
        x = dict(enumerate(row))
        ours.learn_one(x, target)
        theirs.learn_one(x, target)
    assert ours.weights == pytest.approx(theirs.weights, abs=2e-13)
    assert ours.intercept == pytest.approx(theirs.intercept, abs=2e-13)
    assert ours.predict_one(dict(enumerate(X[0]))) == pytest.approx(
        theirs.predict_one(dict(enumerate(X[0]))), abs=2e-13
    )


@pytest.mark.parametrize("mode", [0, 1, 2])
def test_pa_classifier_parity(stream, mode):
    X, target = stream
    y = target > 0
    ours = linear_model.PAClassifier(C=0.08, mode=mode)
    theirs = river_linear.PAClassifier(C=0.08, mode=mode)
    for row, label in zip(X, y):
        x = dict(enumerate(row))
        ours.learn_one(x, label)
        theirs.learn_one(x, label)
    assert ours.weights == pytest.approx(theirs.weights, abs=5e-14)
    assert ours.intercept == pytest.approx(theirs.intercept, abs=5e-14)
    assert ours.predict_proba_one(dict(enumerate(X[0]))) == pytest.approx(
        theirs.predict_proba_one(dict(enumerate(X[0]))), abs=5e-14
    )
    assert ours.predict_one(dict(enumerate(X[0]))) == theirs.predict_one(
        dict(enumerate(X[0]))
    )


def test_sparse_feature_growth_matches_river():
    samples = [
        ({"a": 2.0}, 1.0),
        ({"b": -1.0}, -2.0),
        ({"a": 0.5, "c": 3.0}, 0.3),
        ({"c": -0.2}, 1.4),
    ]
    ours = linear_model.LinearRegression()
    theirs = river_linear.LinearRegression()
    for x, y in samples:
        ours.learn_one(x, y)
        theirs.learn_one(x, y)
    assert ours.weights == pytest.approx(theirs.weights, abs=1e-15)
    assert ours.intercept == pytest.approx(theirs.intercept, abs=1e-15)


def test_covered_constructor_signatures_match_upstream():
    for ours, theirs in [
        (linear_model.LinearRegression, river_linear.LinearRegression),
        (linear_model.LogisticRegression, river_linear.LogisticRegression),
        (linear_model.PARegressor, river_linear.PARegressor),
        (linear_model.PAClassifier, river_linear.PAClassifier),
    ]:
        assert list(inspect.signature(ours).parameters) == list(
            inspect.signature(theirs).parameters
        )


def test_learn_many_rejects_unsafe_shapes_before_ffi():
    X = pd.DataFrame({"x": [1.0, 2.0]})
    model = linear_model.LinearRegression()
    with pytest.raises(ValueError, match="y must contain 2"):
        model.learn_many(X, [1.0])
    with pytest.raises(ValueError, match="w must contain 2"):
        model.learn_many(X, [1.0, 2.0], [1.0])
    with pytest.raises(ValueError, match="one-dimensional"):
        model.learn_many(X, np.ones((2, 1)))
    with pytest.raises(ValueError, match="at least one row"):
        model.learn_many(X.iloc[:0], [])


def test_duplicate_batch_columns_are_rejected():
    X = pd.DataFrame([[1.0, 2.0]], columns=["x", "x"])
    with pytest.raises(ValueError, match="unique"):
        linear_model.LinearRegression().predict_many(X)
