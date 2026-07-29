from __future__ import annotations

import collections

import numpy as np

from ._lib import addr, f64, lib, storage


class StandardScaler:
    def __init__(self, with_std=True) -> None:
        self.with_std = with_std
        self.counts = collections.Counter()
        self.means = collections.defaultdict(float)
        self.vars = collections.defaultdict(float)
        self._features = []
        self._index = {}
        self._counts_array = storage(0)
        self._means_array = storage(0)
        self._vars_array = storage(0)

    def _ensure(self, names):
        added = [name for name in names if name not in self._index]
        if not added:
            return
        old = len(self._features)
        for name in added:
            self._index[name] = len(self._features)
            self._features.append(name)
        size = len(self._features)
        for attr in ("_counts_array", "_means_array", "_vars_array"):
            previous = getattr(self, attr)
            current = storage(size)
            current[:old] = previous[:old]
            setattr(self, attr, current)

    def _sync(self):
        for name, index in self._index.items():
            self.counts[name] = int(self._counts_array[index])
            self.means[name] = float(self._means_array[index])
            if self.with_std:
                self.vars[name] = float(self._vars_array[index])

    def learn_one(self, x):
        self._ensure(x)
        row = np.full(max(1, len(self._features)), np.nan, dtype=np.float64)
        for name, value in x.items():
            row[self._index[name]] = value
        lib().mr_standard_update(
            addr(row), addr(self._counts_array), addr(self._means_array),
            addr(self._vars_array), 1, len(self._features), int(self.with_std)
        )
        self._sync()

    def transform_one(self, x):
        result = {}
        for name, value in x.items():
            centered = value - self.means[name]
            if self.with_std:
                variance = self.vars[name]
                result[name] = centered / variance**0.5 if variance > 0 else centered
            else:
                result[name] = centered
        return result

    def learn_many(self, X):
        if len(X) == 0:
            raise ValueError("learn_many requires at least one row")
        if not X.columns.is_unique:
            raise ValueError("feature columns must be unique")
        self._ensure(X.columns)
        columns = list(X.columns)
        indices = [self._index[name] for name in columns]
        matrix = f64(X.to_numpy())
        aligned = columns == self._features
        if aligned:
            counts = self._counts_array
            means = self._means_array
            variances = self._vars_array
        else:
            counts = f64(self._counts_array[indices])
            means = f64(self._means_array[indices])
            variances = f64(self._vars_array[indices])
        lib().mr_standard_update_many(
            addr(matrix), addr(counts), addr(means), addr(variances),
            len(X), len(columns), int(self.with_std)
        )
        if not aligned:
            self._counts_array[indices] = counts
            self._means_array[indices] = means
            self._vars_array[indices] = variances
        self._sync()

    def transform_many(self, X):
        import pandas as pd

        if not X.columns.is_unique:
            raise ValueError("feature columns must be unique")
        matrix = f64(X.to_numpy())
        columns = list(X.columns)
        if columns == self._features:
            means = self._means_array
            variances = self._vars_array.copy()
        else:
            means = f64([self.means[name] for name in columns], copy=True)
            variances = f64([self.vars[name] for name in columns], copy=True)
        result = np.empty_like(matrix)
        lib().mr_standard_transform(
            addr(matrix), addr(means), addr(variances), addr(result),
            len(X), len(X.columns), int(self.with_std)
        )
        return pd.DataFrame(result, index=X.index, columns=X.columns, copy=False)


class MinMaxScaler:
    def __init__(self):
        self.min = collections.defaultdict(lambda: float("inf"))
        self.max = collections.defaultdict(lambda: float("-inf"))

    def learn_one(self, x):
        for name, value in x.items():
            self.min[name] = min(self.min[name], value)
            self.max[name] = max(self.max[name], value)

    def transform_one(self, x):
        return {
            name: (value - self.min[name]) / (self.max[name] - self.min[name])
            if self.max[name] != self.min[name] else 0.0
            for name, value in x.items()
        }


class MaxAbsScaler:
    def __init__(self):
        self.abs_max = collections.defaultdict(float)

    def learn_one(self, x):
        for name, value in x.items():
            self.abs_max[name] = max(self.abs_max[name], abs(value))

    def transform_one(self, x):
        return {
            name: value / self.abs_max[name] if self.abs_max[name] else 0.0
            for name, value in x.items()
        }


class Normalizer:
    def __init__(self, order=2):
        self.order = order

    def transform_one(self, x):
        if self.order != 2:
            norm = sum(abs(value) ** self.order for value in x.values()) ** (1 / self.order)
            return {name: value / norm for name, value in x.items()}
        names = list(x)
        source = f64([x[name] for name in names], copy=True)
        result = storage(len(names))
        lib().mr_normalize(addr(source), addr(result), 1, len(names))
        return {name: float(result[i]) for i, name in enumerate(names)}

    def transform_many(self, X):
        import pandas as pd

        if self.order != 2:
            values = X.to_numpy(dtype=np.float64)
            norms = np.linalg.norm(values, ord=self.order, axis=1)
            result = values / norms[:, None]
        else:
            values = f64(X.to_numpy())
            result = np.empty_like(values)
            lib().mr_normalize(addr(values), addr(result), len(X), len(X.columns))
        return pd.DataFrame(result, index=X.index, columns=X.columns, copy=False)
