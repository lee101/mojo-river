from __future__ import annotations

import math
import os
import platform
import subprocess
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))

from mojoriver import linear_model as mojo_linear  # noqa: E402
from mojoriver import metrics as mojo_metrics  # noqa: E402
from mojoriver import optim as mojo_optim  # noqa: E402
from mojoriver import preprocessing as mojo_preprocessing  # noqa: E402
from river import linear_model as river_linear  # noqa: E402
from river import metrics as river_metrics  # noqa: E402
from river import optim as river_optim  # noqa: E402
from river import preprocessing as river_preprocessing  # noqa: E402


def time_best(function, repeat=3):
    best = math.inf
    for _ in range(repeat):
        start = time.perf_counter()
        function()
        best = min(best, time.perf_counter() - start)
    return best


def machine_name():
    try:
        line = subprocess.check_output(
            ["lscpu"], text=True, stderr=subprocess.DEVNULL
        )
        for item in line.splitlines():
            if item.startswith("Model name:"):
                return item.split(":", 1)[1].strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return platform.processor() or platform.machine()


CASES = []


def benchmark(name):
    def register(builder):
        CASES.append((name, builder))
        return builder

    return register


rng = np.random.default_rng(123)


@benchmark("LinearRegression.learn_many (200k x 32)")
def linear_many():
    values = rng.normal(size=(200_000, 32))
    X = pd.DataFrame(values, columns=[f"x{i}" for i in range(32)])
    y = pd.Series(values @ rng.normal(size=32) + rng.normal(size=len(X)))

    def ours():
        mojo_linear.LinearRegression(
            optimizer=mojo_optim.SGD(0.001), intercept_lr=0.001
        ).learn_many(X, y)

    def theirs():
        river_linear.LinearRegression(
            optimizer=river_optim.SGD(0.001), intercept_lr=0.001
        ).learn_many(X, y)

    return ours, theirs


@benchmark("LogisticRegression.learn_many (200k x 32)")
def logistic_many():
    values = rng.normal(size=(200_000, 32))
    X = pd.DataFrame(values, columns=[f"x{i}" for i in range(32)])
    y = pd.Series(values[:, 0] - values[:, 1] > 0)

    def ours():
        mojo_linear.LogisticRegression(
            optimizer=mojo_optim.Adam(0.01)
        ).learn_many(X, y)

    def theirs():
        river_linear.LogisticRegression(
            optimizer=river_optim.Adam(0.01)
        ).learn_many(X, y)

    return ours, theirs


@benchmark("LinearRegression.predict_many (500k x 32)")
def linear_predict():
    train = pd.DataFrame(rng.normal(size=(2_000, 32)))
    target = pd.Series(rng.normal(size=len(train)))
    query = pd.DataFrame(rng.normal(size=(500_000, 32)))
    ours_model = mojo_linear.LinearRegression()
    river_model = river_linear.LinearRegression()
    ours_model.learn_many(train, target)
    river_model.learn_many(train, target)
    return (
        lambda: ours_model.predict_many(query),
        lambda: river_model.predict_many(query),
    )


@benchmark("StandardScaler.learn_many (500k x 24)")
def scaler_many():
    X = pd.DataFrame(
        rng.normal(size=(500_000, 24)), columns=[f"x{i}" for i in range(24)]
    )
    return (
        lambda: mojo_preprocessing.StandardScaler().learn_many(X),
        lambda: river_preprocessing.StandardScaler().learn_many(X),
    )


@benchmark("StandardScaler.transform_many (500k x 24)")
def scaler_transform():
    X = pd.DataFrame(
        rng.normal(size=(500_000, 24)), columns=[f"x{i}" for i in range(24)]
    )
    ours_scaler = mojo_preprocessing.StandardScaler()
    river_scaler = river_preprocessing.StandardScaler()
    ours_scaler.learn_many(X[:10_000])
    river_scaler.learn_many(X[:10_000])
    return (
        lambda: ours_scaler.transform_many(X),
        lambda: river_scaler.transform_many(X),
    )


@benchmark("PAClassifier.learn_one stream (25k x 64)")
def pa_stream():
    values = rng.normal(size=(25_000, 64))
    labels = values[:, 0] + values[:, 1] > 0
    stream = [
        ({index: value for index, value in enumerate(row)}, bool(label))
        for row, label in zip(values, labels)
    ]

    def ours():
        model = mojo_linear.PAClassifier(C=0.1)
        for x, y in stream:
            model.learn_one(x, y)

    def theirs():
        model = river_linear.PAClassifier(C=0.1)
        for x, y in stream:
            model.learn_one(x, y)

    return ours, theirs


@benchmark("MSE progressive update (2 million)")
def mse_update():
    truth = rng.normal(size=2_000_000)
    prediction = truth + rng.normal(scale=0.2, size=len(truth))

    def ours():
        metric = mojo_metrics.MSE()
        metric.update_many(truth, prediction)
        return metric.get()

    def theirs():
        metric = river_metrics.MSE()
        for y, pred in zip(truth, prediction):
            metric.update(y, pred)
        return metric.get()

    return ours, theirs


@benchmark("Normalizer L2 (100k x 64)")
def normalizer():
    X = pd.DataFrame(rng.normal(size=(100_000, 64)))
    rows = X.to_dict(orient="records")

    def ours():
        return mojo_preprocessing.Normalizer().transform_many(X)

    def theirs():
        transform = river_preprocessing.Normalizer().transform_one
        return [transform(row) for row in rows]

    return ours, theirs


def main():
    results = []
    for name, builder in CASES:
        ours, theirs = builder()
        ours()
        theirs()
        ours_time = time_best(ours)
        river_time = time_best(theirs)
        results.append((name, ours_time, river_time, river_time / ours_time))

    print(f"Machine: {machine_name()}")
    print(f"Python: {platform.python_version()}, River: 0.23.0")
    print()
    print("| case | mojo-river | river | result |")
    print("| --- | ---: | ---: | ---: |")
    for name, ours_time, river_time, ratio in results:
        comparison = f"{ratio:.2f}x faster" if ratio >= 1 else f"{1 / ratio:.2f}x slower"
        print(
            f"| {name} | {ours_time * 1e3:.2f} ms | "
            f"{river_time * 1e3:.2f} ms | {comparison} |"
        )


if __name__ == "__main__":
    main()
