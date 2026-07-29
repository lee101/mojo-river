from __future__ import annotations

import math

import numpy as np

from ._lib import addr, f64, lib, storage


class Optimizer:
    kind = 0

    def __init__(self, lr):
        if not isinstance(lr, (int, float)):
            raise TypeError("mojoriver optimizers currently require a numeric learning rate")
        self.lr = float(lr)
        self.n_iterations = 0
        self._state1 = storage(0)
        self._state2 = storage(0)

    @property
    def learning_rate(self):
        return self.lr

    def _ensure(self, size):
        if len(self._state1) < max(1, size):
            self._state1 = np.pad(self._state1, (0, max(1, size) - len(self._state1)))
            self._state2 = np.pad(self._state2, (0, max(1, size) - len(self._state2)))

    def _parameters(self):
        return self.lr, 0.0, 0.0, 1e-8

    def _step_array(self, weights, gradient):
        if weights.ndim != 1 or gradient.ndim != 1:
            raise ValueError("array weights and gradients must be one-dimensional")
        if len(weights) != len(gradient):
            raise ValueError("array weights and gradients must have equal length")
        if not weights.flags.writeable:
            raise ValueError("array weights must be writable")
        self._ensure(len(weights))
        lr, beta1, beta2, eps = self._parameters()
        lib().mr_optimizer_step(
            addr(weights),
            addr(gradient),
            addr(self._state1),
            addr(self._state2),
            len(weights),
            self.kind,
            lr,
            beta1,
            beta2,
            eps,
        )
        self.n_iterations += 1

    def step(self, w, g):
        if isinstance(w, np.ndarray) and isinstance(g, np.ndarray):
            gradient = f64(g)
            if w.dtype != np.float64 or not w.flags.c_contiguous:
                raise ValueError("array weights must be C-contiguous float64")
            self._step_array(w, gradient)
            return w
        keys = list(g)
        weights = f64([w.get(key, 0.0) for key in keys], copy=True)
        gradient = f64([g[key] for key in keys], copy=True)
        self._step_array(weights, gradient)
        for key, value in zip(keys, weights):
            w[key] = float(value)
        return w


class SGD(Optimizer):
    def __init__(self, lr=0.01) -> None:
        super().__init__(lr)


class AdaGrad(Optimizer):
    kind = 1

    def __init__(self, lr=0.1, eps=1e-8):
        super().__init__(lr)
        self.eps = eps

    @property
    def g2(self):
        return self._state1

    def _parameters(self):
        return self.lr, 0.0, 0.0, self.eps


class RMSProp(Optimizer):
    kind = 2

    def __init__(self, lr=0.1, rho=0.9, eps=1e-8):
        super().__init__(lr)
        self.rho = rho
        self.eps = eps

    @property
    def g2(self):
        return self._state1

    def _parameters(self):
        return self.lr, self.rho, 0.0, self.eps


class Adam(Optimizer):
    kind = 3

    def __init__(self, lr=0.1, beta_1=0.9, beta_2=0.999, eps=1e-8) -> None:
        super().__init__(lr)
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.eps = eps

    @property
    def m(self):
        return self._state1

    @property
    def v(self):
        return self._state2

    def _parameters(self):
        t = self.n_iterations + 1
        corrected = self.lr * math.sqrt(1 - self.beta_2**t) / (1 - self.beta_1**t)
        return corrected, self.beta_1, self.beta_2, self.eps
