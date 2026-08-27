# mojo-river

`mojo-river` is a Mojo implementation of the compute-heavy inner loops in
[River](https://riverml.xyz/), exposed to Python with River's online-learning
names, signatures, and stateful behavior.

It is not a reimplementation of all of River. The covered subset is intended
for dense numeric streams and mini-batches:

| River module | Covered API |
| --- | --- |
| `linear_model` | `LinearRegression`, `LogisticRegression`, `PARegressor`, `PAClassifier`; `learn_one` and single-item prediction for all four, plus `learn_many`, `predict_many`, and probability prediction for the two GLMs |
| `optim` | `SGD`, `AdaGrad`, `RMSProp`, `Adam`, including dictionary and NumPy-array `step` |
| `preprocessing` | `StandardScaler` (`learn_one`, `transform_one`, `learn_many`, `transform_many`); `MinMaxScaler` and `MaxAbsScaler` (`learn_one`, `transform_one`); `Normalizer` (`transform_one`, `transform_many`) |
| `metrics` | `MAE`, `MSE`, `RMSE`, `LogLoss`, `Accuracy`; weighted `update`, `revert`, and a Mojo-backed `update_many` extension |

Every method listed above has a test, and its numerical behavior is parity-tested
against River 0.23.0. The Mojo
side implements the dense arithmetic; the small `MinMaxScaler` and
`MaxAbsScaler` dictionary updates remain in Python because an FFI call would
cost more than their work.

Not covered are trees, ensembles, clustering, anomaly detection, time-series
models, multiclass linear models, custom losses or initializers, L1
regularization, optimizer schedulers, shared metric confusion matrices, and
River's pipeline/evaluation/dataset infrastructure. Inputs may use arbitrary
hashable feature names, but the performance-oriented path assumes a stable
dense schema.

## Install

The repository carries a pinned Mojo nightly and the real upstream River
package in its Pixi environment:

```bash
pixi install
pixi run build
pixi run test
```

`pixi run build` produces `dist/libmojo-river.so`. Importing `mojoriver` also
rebuilds the library when the Mojo source is newer than the shared object.

## Usage

The API follows River, with `mojoriver` replacing the top-level import:

```python
from mojoriver import linear_model, metrics, optim, preprocessing

model = linear_model.LogisticRegression(optimizer=optim.Adam(0.03))
scaler = preprocessing.StandardScaler()
accuracy = metrics.Accuracy()

stream = [
    ({"price": 10.0, "volume": 2.0}, False),
    ({"price": 12.0, "volume": 8.0}, True),
    ({"price": 11.0, "volume": 6.0}, True),
]

for x, y in stream:
    scaler.learn_one(x)
    x_scaled = scaler.transform_one(x)
    prediction = model.predict_one(x_scaled)
    accuracy.update(y, prediction)
    model.learn_one(x_scaled, y)

print(model.predict_proba_one({"price": 0.5, "volume": 1.0}))
print(accuracy.get())
```

## Performance

Measured with `pixi run bench` on an Intel Xeon E5-2697 v4 at 2.30 GHz, using
Python 3.13.14 and River 0.23.0. Times are the best of three runs after one
warm-up. Stateful learning cases construct a fresh model inside every timed
run.

| case | mojo-river | river | result |
| --- | ---: | ---: | ---: |
| LinearRegression.learn_many (200k x 32) | 12.05 ms | 93.31 ms | 7.74x faster |
| LogisticRegression.learn_many (200k x 32) | 18.34 ms | 59.56 ms | 3.25x faster |
| LinearRegression.predict_many (500k x 32) | 11.68 ms | 11.99 ms | 1.03x faster |
| StandardScaler.learn_many (500k x 24) | 51.70 ms | 192.67 ms | 3.73x faster |
| StandardScaler.transform_many (500k x 24) | 48.63 ms | 104.79 ms | 2.15x faster |
| PAClassifier.learn_one stream (25k x 64) | 587.12 ms | 1085.14 ms | 1.85x faster |
| MSE progressive update (2 million) | 6.83 ms | 1507.66 ms | 220.70x faster |
| Normalizer L2 (100k x 64) | 62.15 ms | 2032.96 ms | 32.71x faster |

The metric and normalizer comparisons use `update_many` and `transform_many`
to move a complete dense buffer in one call; upstream River exposes only
per-item methods for those classes. The prediction result and all other
changes in the table are reported as measured.

No GPU path is included. Dense prediction performs roughly 0.25 floating-point
operations per byte read, and the other kernels are similarly memory-bound, so
this port keeps computation on the CPU and parallelizes only large independent
batch workloads.

Run the benchmark only through Pixi so the machine-wide benchmark lock is
held:

```bash
pixi run bench
```

## How it works

All kernels live in one Mojo compilation unit to avoid repeated compiler
startup cost. Python owns model state and NumPy storage. A ctypes call passes
each contiguous `float64` buffer as an integer address; the exported Mojo C ABI
function reconstructs a mutable typed pointer, operates directly on row-major
memory, and returns scalar state such as the updated intercept.

The FFI performs no allocation and retains no pointers. Weight, gradient,
optimizer-moment, scaler-statistic, input, and result buffers all remain owned
by Python. Stable DataFrame column order takes the zero-copy fast path;
changing schemas are remapped once into the model's persistent feature order.
Dot products and scaler kernels use SIMD with explicit scalar remainder loops.
Prediction evaluates four rows per SIMD loop so weight vectors are reused and
independent accumulators can overlap. Batches with at least one million numeric
values split contiguous row ranges across at most eight persistent host workers;
smaller batches stay serial. Each worker passes a zero-copy NumPy slice through
the FFI. When a DataFrame already matches the scaler's persistent schema,
statistic buffers also cross the FFI directly without temporary copies.

## License

MIT
