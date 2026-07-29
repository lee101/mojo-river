from __future__ import annotations

import math

import numpy as np

from ._lib import addr, lib, vector


class _MeanMetric:
    kernel = ""

    def __init__(self):
        self._sum = 0.0
        self._weight = 0.0

    def _eval(self, y_true, y_pred):
        raise NotImplementedError

    def update(self, y_true, y_pred, w=1.0) -> None:
        self._sum += float(w) * self._eval(y_true, y_pred)
        self._weight += float(w)

    def update_many(self, y_true, y_pred, w=1.0):
        truth = vector(y_true, name="y_true")
        prediction = vector(y_pred, length=len(truth), name="y_pred")
        weights = (
            np.full(len(truth), float(w), dtype=np.float64)
            if np.isscalar(w)
            else vector(w, length=len(truth), name="w")
        )
        self._sum += getattr(lib(), self.kernel)(
            addr(truth), addr(prediction), addr(weights), len(truth)
        )
        self._weight += float(weights.sum())

    def revert(self, y_true, y_pred, w=1.0) -> None:
        self._sum -= float(w) * self._eval(y_true, y_pred)
        self._weight -= float(w)

    def get(self):
        return self._sum / self._weight if self._weight else 0.0


class MAE(_MeanMetric):
    kernel = "mr_sum_absolute_error"

    def _eval(self, y_true, y_pred):
        return abs(y_true - y_pred)


class MSE(_MeanMetric):
    kernel = "mr_sum_squared_error"

    def _eval(self, y_true, y_pred):
        return (y_true - y_pred) ** 2


class RMSE(MSE):
    def get(self):
        return super().get() ** 0.5


class LogLoss(_MeanMetric):
    kernel = "mr_sum_log_loss"

    def _eval(self, y_true, y_pred):
        probability = y_pred.get(True, 0.0) if isinstance(y_pred, dict) else y_pred
        probability = min(max(probability, 1e-15), 1 - 1e-15)
        return -math.log(probability if y_true else 1 - probability)


class Accuracy(_MeanMetric):
    kernel = "mr_sum_accuracy"

    def __init__(self, cm=None):
        if cm is not None:
            raise NotImplementedError("shared confusion matrices are not in the covered subset")
        super().__init__()
        self.cm = cm

    def _eval(self, y_true, y_pred):
        return float(y_true == y_pred)
