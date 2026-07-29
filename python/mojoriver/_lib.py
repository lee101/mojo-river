from __future__ import annotations

import ctypes
import os
import subprocess

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LIB_PATH = os.path.join(ROOT, "dist", "libmojo-river.so")

I = ctypes.c_int64
F = ctypes.c_double

_SIGNATURES = {
    "mr_linear_learn_one": ([I] * 6 + [F] * 6 + [I] + [F] * 4, F),
    "mr_logistic_learn_one": ([I] * 6 + [F] * 6 + [I] + [F] * 4, F),
    "mr_linear_learn_many": ([I] * 9 + [F] * 4 + [I] + [F] * 4, F),
    "mr_logistic_learn_many": ([I] * 9 + [F] * 4 + [I] + [F] * 4, F),
    "mr_predict": ([I, I, I, I, I, F, I], None),
    "mr_optimizer_step": ([I, I, I, I, I, I, F, F, F, F], None),
    "mr_pa_regression_learn": ([I, I, I, F, F, F, I, F, I], F),
    "mr_pa_classification_learn": ([I, I, I, I, F, F, I, I], F),
    "mr_standard_update": ([I] * 7, None),
    "mr_standard_update_many": ([I] * 7, None),
    "mr_standard_transform": ([I] * 7, None),
    "mr_normalize": ([I] * 4, None),
    "mr_sum_absolute_error": ([I] * 4, F),
    "mr_sum_squared_error": ([I] * 4, F),
    "mr_sum_log_loss": ([I] * 4, F),
    "mr_sum_accuracy": ([I] * 4, F),
}

_library = None


def build() -> str:
    source = os.path.join(ROOT, "src", "kernels.mojo")
    if not os.path.exists(LIB_PATH) or os.path.getmtime(LIB_PATH) < os.path.getmtime(source):
        subprocess.run(
            ["bash", os.path.join(ROOT, "build", "build.sh")],
            cwd=ROOT,
            check=True,
        )
    return LIB_PATH


def lib() -> ctypes.CDLL:
    global _library
    if _library is None:
        _library = ctypes.CDLL(build())
        for name, (argtypes, restype) in _SIGNATURES.items():
            fn = getattr(_library, name)
            fn.argtypes = argtypes
            fn.restype = restype
    return _library


def f64(values, *, copy: bool = False) -> np.ndarray:
    source = np.asarray(values)
    if source.dtype.kind not in "biuf":
        raise TypeError("Mojo kernels require real numeric input")
    if source.dtype.kind == "f" and source.dtype.itemsize > np.dtype(np.float64).itemsize:
        raise ValueError("input precision exceeds the float64 kernel dtype")
    if source.dtype.kind in "iu" and source.size:
        exact_limit = 1 << 53
        if np.any(source > exact_limit) or (
            source.dtype.kind == "i" and np.any(source < -exact_limit)
        ):
            raise ValueError("integer input cannot be represented exactly as float64")
    if copy:
        return np.array(source, dtype=np.float64, order="C", copy=True)
    return np.ascontiguousarray(source, dtype=np.float64)


def addr(array: np.ndarray) -> int:
    if not isinstance(array, np.ndarray):
        raise TypeError("FFI buffers must be NumPy arrays")
    if array.dtype != np.float64 or not array.flags.c_contiguous:
        raise ValueError("FFI buffers must be C-contiguous float64 arrays")
    address = int(array.ctypes.data)
    if address == 0:
        raise ValueError("FFI buffers must have a non-null address")
    return address


def vector(values, *, length: int | None = None, name: str = "array") -> np.ndarray:
    result = f64(values)
    if result.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if length is not None and len(result) != length:
        raise ValueError(f"{name} must contain {length} values, got {len(result)}")
    return result


def storage(size: int) -> np.ndarray:
    return np.zeros(max(1, size), dtype=np.float64)
