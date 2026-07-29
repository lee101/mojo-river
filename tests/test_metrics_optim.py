import numpy as np
import pytest

from river import metrics as river_metrics
from river import optim as river_optim

from mojoriver import metrics, optim


@pytest.mark.parametrize(
    ("ours_type", "river_type"),
    [
        (metrics.MAE, river_metrics.MAE),
        (metrics.MSE, river_metrics.MSE),
        (metrics.RMSE, river_metrics.RMSE),
        (metrics.LogLoss, river_metrics.LogLoss),
        (metrics.Accuracy, river_metrics.Accuracy),
    ],
)
def test_metrics_update_and_revert_parity(ours_type, river_type):
    truth = [True, False, True, False] if ours_type is metrics.Accuracy else [3.0, -0.5, 2.0, 7.0]
    if ours_type is metrics.LogLoss:
        prediction = [0.9, 0.1, 0.7, 0.35]
    elif ours_type is metrics.Accuracy:
        prediction = [True, True, True, False]
    else:
        prediction = [2.5, 0.0, 2.0, 8.0]
    ours = ours_type()
    theirs = river_type()
    for y, pred, weight in zip(truth, prediction, [1.0, 2.0, 0.5, 1.5]):
        ours.update(y, pred, weight)
        theirs.update(y, pred, weight)
        assert ours.get() == pytest.approx(theirs.get())
    ours.revert(truth[-1], prediction[-1], 1.5)
    theirs.revert(truth[-1], prediction[-1], 1.5)
    assert ours.get() == pytest.approx(theirs.get())


@pytest.mark.parametrize(
    ("ours_type", "river_type", "prediction"),
    [
        (metrics.MAE, river_metrics.MAE, None),
        (metrics.MSE, river_metrics.MSE, None),
        (metrics.RMSE, river_metrics.RMSE, None),
        (metrics.LogLoss, river_metrics.LogLoss, "probability"),
        (metrics.Accuracy, river_metrics.Accuracy, "labels"),
    ],
)
def test_metric_mojo_batch_extension_matches_river(ours_type, river_type, prediction):
    rng = np.random.default_rng(11)
    truth = rng.normal(size=500)
    pred = rng.normal(size=500)
    if prediction == "probability":
        truth = truth > 0
        pred = 1 / (1 + np.exp(-pred))
    elif prediction == "labels":
        truth = truth > 0
        pred = pred > 0
    weights = rng.uniform(0.2, 2.0, size=500)
    ours = ours_type()
    ours.update_many(truth, pred, weights)
    theirs = river_type()
    for y, p, w in zip(truth, pred, weights):
        theirs.update(y.item(), p.item(), w.item())
    tolerance = 3e-11 if ours_type is metrics.LogLoss else 2e-14
    assert ours.get() == pytest.approx(theirs.get(), abs=tolerance)


@pytest.mark.parametrize(
    ("ours_optimizer", "river_optimizer"),
    [
        (optim.SGD(0.03), river_optim.SGD(0.03)),
        (optim.AdaGrad(0.03), river_optim.AdaGrad(0.03)),
        (optim.RMSProp(0.03), river_optim.RMSProp(0.03)),
        (optim.Adam(0.03), river_optim.Adam(0.03)),
    ],
)
def test_optimizer_step_array_parity(ours_optimizer, river_optimizer):
    ours_weights = {"a": 0.2, "b": -0.5, "c": 1.0}
    river_weights = ours_weights.copy()
    gradients = [
        {"a": 0.3, "b": -0.1, "c": 0.7},
        {"a": -0.2, "b": 0.4, "c": 0.1},
        {"a": 0.05, "b": -0.3, "c": 0.9},
    ]
    for gradient in gradients:
        ours_optimizer.step(ours_weights, gradient)
        river_optimizer.step(river_weights, gradient)
    assert ours_weights == pytest.approx(river_weights, abs=2e-14)
    assert ours_optimizer.n_iterations == river_optimizer.n_iterations


@pytest.mark.parametrize(
    ("ours_optimizer", "river_optimizer"),
    [
        (optim.SGD(0.03), river_optim.SGD(0.03)),
        (optim.AdaGrad(0.03), river_optim.AdaGrad(0.03)),
        (optim.RMSProp(0.03), river_optim.RMSProp(0.03)),
        (optim.Adam(0.03), river_optim.Adam(0.03)),
    ],
)
def test_optimizer_direct_numpy_step(ours_optimizer, river_optimizer):
    weights = np.array([0.2, -0.5, 1.0], dtype=np.float64)
    gradient = np.array([0.3, -0.1, 0.7], dtype=np.float64)
    expected = {"a": 0.2, "b": -0.5, "c": 1.0}
    river_optimizer.step(expected, dict(zip(expected, gradient)))
    result = ours_optimizer.step(weights, gradient)
    assert result is weights
    assert weights == pytest.approx(list(expected.values()))


@pytest.mark.parametrize(
    ("weights", "gradient", "message"),
    [
        (np.ones(2, dtype=np.float64), np.ones(1), "equal length"),
        (np.ones((1, 2), dtype=np.float64), np.ones(2), "one-dimensional"),
        (np.ones(2, dtype=np.float32), np.ones(2), "C-contiguous float64"),
    ],
)
def test_optimizer_rejects_unsafe_array_inputs(weights, gradient, message):
    with pytest.raises(ValueError, match=message):
        optim.SGD().step(weights, gradient)


def test_metric_batch_rejects_mismatched_or_nonnumeric_buffers():
    metric = metrics.MSE()
    with pytest.raises(ValueError, match="y_pred must contain 2"):
        metric.update_many([1.0, 2.0], [1.0])
    with pytest.raises(ValueError, match="w must contain 2"):
        metric.update_many([1.0, 2.0], [1.0, 2.0], [1.0])
    with pytest.raises(TypeError, match="real numeric"):
        metric.update_many(["one"], [1.0])
    with pytest.raises(ValueError, match="represented exactly"):
        metric.update_many([2**53 + 1], [1.0])
