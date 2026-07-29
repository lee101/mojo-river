import numpy as np
import pandas as pd
import pytest

from river import preprocessing as river_preprocessing

from mojoriver import preprocessing


def records():
    rng = np.random.default_rng(9)
    rows = rng.normal(size=(30, 4))
    return [{name: value for name, value in zip("abcd", row)} for row in rows]


@pytest.mark.parametrize("with_std", [False, True])
def test_standard_scaler_online_parity(with_std):
    ours = preprocessing.StandardScaler(with_std=with_std)
    theirs = river_preprocessing.StandardScaler(with_std=with_std)
    for x in records():
        ours.learn_one(x)
        theirs.learn_one(x)
        assert ours.transform_one(x) == pytest.approx(theirs.transform_one(x), abs=1e-14)
    assert ours.counts == theirs.counts
    assert ours.means == pytest.approx(theirs.means, abs=1e-14)
    assert ours.vars == pytest.approx(theirs.vars, abs=1e-14)


def test_standard_scaler_sparse_feature_updates_match_river():
    samples = [{"a": 1.0}, {"b": 4.0}, {"a": 3.0}, {"a": 2.0, "b": 8.0}]
    ours = preprocessing.StandardScaler()
    theirs = river_preprocessing.StandardScaler()
    for x in samples:
        ours.learn_one(x)
        theirs.learn_one(x)
    assert ours.counts == theirs.counts
    assert ours.means == pytest.approx(theirs.means)
    assert ours.vars == pytest.approx(theirs.vars)


def test_standard_scaler_minibatch_parity_with_nan():
    X = pd.DataFrame(records())
    X.loc[::7, "b"] = np.nan
    ours = preprocessing.StandardScaler()
    theirs = river_preprocessing.StandardScaler()
    for indices in np.array_split(np.arange(len(X)), 3):
        batch = X.iloc[indices]
        ours.learn_many(batch)
        theirs.learn_many(batch)
    assert ours.counts == theirs.counts
    assert ours.means == pytest.approx(theirs.means, abs=2e-14)
    for name in X:
        if np.isnan(theirs.vars[name]):
            assert np.isnan(ours.vars[name])
        else:
            assert ours.vars[name] == pytest.approx(theirs.vars[name], abs=2e-14)
    assert np.allclose(
        ours.transform_many(X).to_numpy(),
        theirs.transform_many(X).to_numpy(),
        equal_nan=True,
    )


@pytest.mark.parametrize("rows", [100, 16_000], ids=["serial", "parallel"])
@pytest.mark.parametrize("with_std", [False, True])
def test_standard_scaler_simd_tail_across_parallel_threshold(rows, with_std):
    rng = np.random.default_rng(21)
    values = rng.normal(size=(rows, 17))
    values[:, 0] = 3.0
    values[::997, 1] = np.nan
    X = pd.DataFrame(values)
    ours = preprocessing.StandardScaler(with_std=with_std)
    theirs = river_preprocessing.StandardScaler(with_std=with_std)

    ours.learn_many(X)
    theirs.learn_many(X)

    assert ours.counts == theirs.counts
    assert ours.means == pytest.approx(theirs.means, abs=2e-14)
    for name in X:
        if np.isnan(theirs.vars[name]):
            assert np.isnan(ours.vars[name])
        else:
            assert ours.vars[name] == pytest.approx(theirs.vars[name], abs=2e-14)
    assert np.allclose(
        ours.transform_many(X).to_numpy(),
        theirs.transform_many(X).to_numpy(),
        rtol=1e-12,
        atol=2e-14,
        equal_nan=True,
    )


def test_minmax_scaler_parity():
    ours = preprocessing.MinMaxScaler()
    theirs = river_preprocessing.MinMaxScaler()
    for x in records():
        ours.learn_one(x)
        theirs.learn_one(x)
        assert ours.transform_one(x) == pytest.approx(theirs.transform_one(x))


def test_maxabs_scaler_parity():
    ours = preprocessing.MaxAbsScaler()
    theirs = river_preprocessing.MaxAbsScaler()
    for x in records():
        ours.learn_one(x)
        theirs.learn_one(x)
        assert ours.transform_one(x) == pytest.approx(theirs.transform_one(x))


@pytest.mark.parametrize("order", [1, 2, 3])
def test_normalizer_parity(order):
    ours = preprocessing.Normalizer(order=order)
    theirs = river_preprocessing.Normalizer(order=order)
    for x in records():
        assert ours.transform_one(x) == pytest.approx(theirs.transform_one(x), abs=1e-14)


def test_normalizer_many_matches_rowwise_river():
    X = pd.DataFrame(records())
    ours = preprocessing.Normalizer().transform_many(X)
    expected = pd.DataFrame(
        [river_preprocessing.Normalizer().transform_one(row) for row in records()]
    )
    assert np.allclose(ours, expected)


def test_standard_scaler_rejects_empty_and_duplicate_batches():
    scaler = preprocessing.StandardScaler()
    with pytest.raises(ValueError, match="at least one row"):
        scaler.learn_many(pd.DataFrame(columns=["x"]))
    duplicate = pd.DataFrame([[1.0, 2.0]], columns=["x", "x"])
    with pytest.raises(ValueError, match="unique"):
        scaler.learn_many(duplicate)
    with pytest.raises(ValueError, match="unique"):
        scaler.transform_many(duplicate)
