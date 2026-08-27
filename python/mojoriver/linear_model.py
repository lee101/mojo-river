from __future__ import annotations

import numbers
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from . import optim
from ._lib import addr, f64, lib, storage, vector


_PREDICT_PARALLEL_WORK = 1_000_000
_PREDICT_MAX_WORKERS = 8
_executor = None


def _prediction_executor():
    global _executor
    if _executor is None:
        _executor = ThreadPoolExecutor(
            max_workers=min(_PREDICT_MAX_WORKERS, os.cpu_count() or 1)
        )
    return _executor


class _DenseFeatures:
    def __init__(self):
        self._features = []
        self._feature_index = {}
        self._weights_array = storage(0)

    def _ensure_features(self, names):
        added = [name for name in names if name not in self._feature_index]
        if not added:
            return
        old_size = len(self._features)
        for name in added:
            self._feature_index[name] = len(self._features)
            self._features.append(name)
        new_weights = storage(len(self._features))
        new_weights[:old_size] = self._weights_array[:old_size]
        self._weights_array = new_weights
        self._resized()

    def _resized(self):
        pass

    def _vector(self, x, add=False):
        if add:
            self._ensure_features(x)
        vector = storage(len(self._features))
        for name, value in x.items():
            index = self._feature_index.get(name)
            if index is not None:
                vector[index] = value
        return vector

    def _matrix(self, X, add=False):
        columns = list(X.columns)
        if len(set(columns)) != len(columns):
            raise ValueError("feature columns must be unique")
        if add:
            self._ensure_features(columns)
        if columns == self._features:
            return f64(X.to_numpy())
        matrix = np.zeros((len(X), max(1, len(self._features))), dtype=np.float64)
        for column in columns:
            index = self._feature_index.get(column)
            if index is not None:
                matrix[:, index] = np.asarray(X[column], dtype=np.float64)
        return np.ascontiguousarray(matrix[:, : len(self._features)] if self._features else matrix)


class _GLM(_DenseFeatures):
    _logistic = False

    def __init__(
        self,
        optimizer,
        loss,
        l2,
        l1,
        intercept_init,
        intercept_lr,
        clip_gradient,
        initializer,
    ):
        super().__init__()
        if loss is not None:
            raise NotImplementedError("custom loss functions are not in the covered subset")
        if l1 != 0:
            raise NotImplementedError("L1 regularization is not in the covered subset")
        if initializer is not None:
            raise NotImplementedError("custom initializers are not in the covered subset")
        self.optimizer = optim.SGD(0.01) if optimizer is None else optimizer
        if not isinstance(self.optimizer, optim.Optimizer):
            raise TypeError("optimizer must come from mojoriver.optim")
        self.loss = loss
        self.l2 = l2
        self.l1 = l1
        self.intercept_init = intercept_init
        self.intercept = float(intercept_init)
        self.intercept_lr = intercept_lr
        self.clip_gradient = clip_gradient
        self.initializer = initializer
        self._gradient = storage(0)
        self._y_name = None

    def _resized(self):
        self._gradient = storage(len(self._features))
        self.optimizer._ensure(len(self._features))

    @property
    def weights(self):
        return {
            feature: float(self._weights_array[index])
            for feature, index in self._feature_index.items()
        }

    def _intercept_rate(self):
        if isinstance(self.intercept_lr, numbers.Number):
            return float(self.intercept_lr)
        return float(self.intercept_lr.get(self.optimizer.n_iterations))

    def _optimizer_parameters(self):
        return self.optimizer._parameters()

    def learn_one(self, x, y, w=1.0) -> None:
        vector = self._vector(x, add=True)
        rate, beta1, beta2, eps = self._optimizer_parameters()
        function = (
            lib().mr_logistic_learn_one
            if self._logistic
            else lib().mr_linear_learn_one
        )
        self.intercept = function(
            addr(vector),
            addr(self._weights_array),
            addr(self._gradient),
            addr(self.optimizer._state1),
            addr(self.optimizer._state2),
            len(self._features),
            float(y),
            float(w),
            self.intercept,
            self._intercept_rate(),
            self.l2,
            self.clip_gradient,
            self.optimizer.kind,
            rate,
            beta1,
            beta2,
            eps,
        )
        self.optimizer.n_iterations += 1

    def learn_many(self, X, y, w=1) -> None:
        if len(X) == 0:
            raise ValueError("learn_many requires at least one row")
        matrix = self._matrix(X, add=True)
        target = vector(y, length=len(X), name="y")
        sample_weights = (
            np.full(len(X), float(w), dtype=np.float64)
            if isinstance(w, numbers.Number)
            else vector(w, length=len(X), name="w")
        )
        rate, beta1, beta2, eps = self._optimizer_parameters()
        function = (
            lib().mr_logistic_learn_many
            if self._logistic
            else lib().mr_linear_learn_many
        )
        self.intercept = function(
            addr(matrix),
            addr(target),
            addr(sample_weights),
            addr(self._weights_array),
            addr(self._gradient),
            addr(self.optimizer._state1),
            addr(self.optimizer._state2),
            len(X),
            len(self._features),
            self.intercept,
            self._intercept_rate(),
            self.l2,
            self.clip_gradient,
            self.optimizer.kind,
            rate,
            beta1,
            beta2,
            eps,
        )
        self.optimizer.n_iterations += 1
        self._y_name = getattr(y, "name", None)

    def _predict_array(self, matrix):
        result = np.empty(len(matrix), dtype=np.float64)
        library = lib()
        if matrix.size >= _PREDICT_PARALLEL_WORK and len(matrix) > 1:
            workers = min(
                _PREDICT_MAX_WORKERS,
                os.cpu_count() or 1,
                (matrix.size + _PREDICT_PARALLEL_WORK - 1)
                // _PREDICT_PARALLEL_WORK,
            )
            chunk = (len(matrix) + workers - 1) // workers
            futures = []
            for start in range(0, len(matrix), chunk):
                stop = min(start + chunk, len(matrix))
                matrix_chunk = matrix[start:stop]
                result_chunk = result[start:stop]
                futures.append(
                    _prediction_executor().submit(
                        library.mr_predict,
                        addr(matrix_chunk),
                        addr(self._weights_array),
                        addr(result_chunk),
                        len(matrix_chunk),
                        len(self._features),
                        self.intercept,
                        int(self._logistic),
                    )
                )
            for future in futures:
                future.result()
        else:
            library.mr_predict(
                addr(matrix),
                addr(self._weights_array),
                addr(result),
                len(matrix),
                len(self._features),
                self.intercept,
                int(self._logistic),
            )
        return result

    def _raw_or_probability_one(self, x):
        matrix = self._vector(x).reshape(1, -1)
        return float(self._predict_array(matrix)[0])


class LinearRegression(_GLM):
    def __init__(
        self,
        optimizer: optim.Optimizer | None = None,
        loss=None,
        l2=0.0,
        l1=0.0,
        intercept_init=0.0,
        intercept_lr=0.01,
        clip_gradient=1e12,
        initializer=None,
    ):
        super().__init__(
            optimizer, loss, l2, l1, intercept_init, intercept_lr, clip_gradient, initializer
        )

    def predict_one(self, x):
        return self._raw_or_probability_one(x)

    def predict_many(self, X):
        import pandas as pd

        values = self._predict_array(self._matrix(X))
        return pd.Series(values, index=X.index, name=self._y_name, copy=False)


class LogisticRegression(_GLM):
    _logistic = True

    def __init__(
        self,
        optimizer: optim.Optimizer | None = None,
        loss=None,
        l2=0.0,
        l1=0.0,
        intercept_init=0.0,
        intercept_lr=0.01,
        clip_gradient=1e12,
        initializer=None,
    ):
        super().__init__(
            optimizer, loss, l2, l1, intercept_init, intercept_lr, clip_gradient, initializer
        )

    def predict_proba_one(self, x):
        probability = self._raw_or_probability_one(x)
        return {False: 1.0 - probability, True: probability}

    def predict_one(self, x):
        probabilities = self.predict_proba_one(x)
        return probabilities[True] > probabilities[False]

    def predict_proba_many(self, X):
        import pandas as pd

        probability = self._predict_array(self._matrix(X))
        return pd.DataFrame(
            {False: 1.0 - probability, True: probability},
            index=X.index,
            copy=False,
        )

    def predict_many(self, X):
        return self.predict_proba_many(X).idxmax(axis="columns")


class BasePA(_DenseFeatures):
    def __init__(self, C, mode, learn_intercept):
        super().__init__()
        if mode not in (0, 1, 2):
            raise ValueError("mode must be 0, 1, or 2")
        self.C = C
        self.mode = mode
        self.learn_intercept = learn_intercept
        self.intercept = 0.0

    @property
    def weights(self):
        return {
            feature: float(self._weights_array[index])
            for feature, index in self._feature_index.items()
        }

    def _raw(self, x):
        vector = self._vector(x)
        result = np.empty(1, dtype=np.float64)
        lib().mr_predict(
            addr(vector), addr(self._weights_array), addr(result), 1,
            len(self._features), self.intercept, 0
        )
        return float(result[0])


class PARegressor(BasePA):
    def __init__(self, C=1.0, mode=1, eps=0.1, learn_intercept=True):
        super().__init__(C=C, mode=mode, learn_intercept=learn_intercept)
        self.eps = eps

    def learn_one(self, x, y):
        vector = self._vector(x, add=True)
        self.intercept = lib().mr_pa_regression_learn(
            addr(vector), addr(self._weights_array), len(self._features), float(y),
            self.intercept, self.C, self.mode, self.eps, int(self.learn_intercept)
        )

    def predict_one(self, x):
        return self._raw(x)


class PAClassifier(BasePA):
    def __init__(self, C=1.0, mode=1, learn_intercept=True):
        super().__init__(C=C, mode=mode, learn_intercept=learn_intercept)

    def learn_one(self, x, y):
        vector = self._vector(x, add=True)
        self.intercept = lib().mr_pa_classification_learn(
            addr(vector), addr(self._weights_array), len(self._features), int(bool(y)),
            self.intercept, self.C, self.mode, int(self.learn_intercept)
        )

    def predict_proba_one(self, x):
        raw = self._raw(x)
        if raw >= 0:
            probability = 1.0 / (1.0 + np.exp(-raw))
        else:
            exp_raw = np.exp(raw)
            probability = exp_raw / (1.0 + exp_raw)
        return {False: 1.0 - probability, True: probability}

    def predict_one(self, x):
        probability = self.predict_proba_one(x)
        return probability[True] > probability[False]
