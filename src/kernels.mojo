"""Dense online-learning kernels exposed through a small C ABI."""

from std.math import exp, log, sqrt
from std.sys.info import simd_width_of as simdwidthof


comptime Ptr = Pointer[Float64, AnyOrigin[mut=True]]


def p(addr: Int) -> Ptr:
    return Ptr(unsafe_from_address=addr)


def dot(x: Ptr, w: Ptr, d: Int) -> Float64:
    comptime W = 4
    var acc = SIMD[DType.float64, W](0.0)
    var i = 0
    while i + W <= d:
        acc += x.load[width=W](i) * w.load[width=W](i)
        i += W
    var total = acc.reduce_add()
    while i < d:
        total += x[i] * w[i]
        i += 1
    return total


def standard_update_vector(
    x: Ptr,
    counts: Ptr,
    means: Ptr,
    variances: Ptr,
    n: Int,
    d: Int,
    j: Int,
    with_std: Bool,
):
    comptime W = simdwidthof[DType.float64]()
    var new_count = SIMD[DType.float64, W](0.0)
    var new_sum = SIMD[DType.float64, W](0.0)
    var square_sum = SIMD[DType.float64, W](0.0)
    var i = 0
    while i < n:
        var values = x.load[width=W](i * d + j)
        square_sum += values * values
        var valid = values.eq(values)
        new_sum += valid.select(values, SIMD[DType.float64, W](0.0))
        new_count += valid.select(
            SIMD[DType.float64, W](1.0), SIMD[DType.float64, W](0.0)
        )
        i += 1

    var old_count = counts.load[width=W](j)
    var combined = old_count + new_count
    var valid_count = new_count.gt(0.0)
    var safe_count = valid_count.select(new_count, SIMD[DType.float64, W](1.0))
    var safe_combined = valid_count.select(combined, SIMD[DType.float64, W](1.0))
    var new_mean = new_sum / safe_count
    var a = old_count / safe_combined
    var b = new_count / safe_combined
    var old_mean = means.load[width=W](j)
    means.store(j, valid_count.select(a * old_mean + b * new_mean, old_mean))
    if with_std:
        var old_variance = variances.load[width=W](j)
        var new_variance = square_sum / Float64(n) - new_mean * new_mean
        var difference = old_mean - new_mean
        variances.store(
            j,
            valid_count.select(
                a * old_variance
                + b * new_variance
                + a * b * difference * difference,
                old_variance,
            ),
        )
    counts.store(j, valid_count.select(combined, old_count))


def standard_update_scalar(
    x: Ptr,
    counts: Ptr,
    means: Ptr,
    variances: Ptr,
    n: Int,
    d: Int,
    j: Int,
    with_std: Bool,
):
    var new_count = 0.0
    var new_sum = 0.0
    var square_sum = 0.0
    for i in range(n):
        var value = x[i * d + j]
        square_sum += value * value
        if value == value:
            new_sum += value
            new_count += 1.0
    if new_count == 0.0:
        return
    var new_mean = new_sum / new_count
    var old_count = counts[j]
    var combined = old_count + new_count
    var a = old_count / combined
    var b = new_count / combined
    var old_mean = means[j]
    means[j] = a * old_mean + b * new_mean
    if with_std:
        var new_variance = square_sum / Float64(n) - new_mean * new_mean
        var difference = old_mean - new_mean
        variances[j] = (
            a * variances[j]
            + b * new_variance
            + a * b * difference * difference
        )
    counts[j] = combined


def standard_transform_rows(
    x: Ptr,
    means: Ptr,
    scales: Ptr,
    destination: Ptr,
    d: Int,
    start: Int,
    end: Int,
):
    comptime W = simdwidthof[DType.float64]()
    var full = d - d % W
    for i in range(start, end):
        var offset = i * d
        var j = 0
        while j < full:
            var values = (
                x.load[width=W](offset + j) - means.load[width=W](j)
            ) * scales.load[width=W](j)
            destination.store(offset + j, values)
            j += W
        while j < d:
            destination[offset + j] = (x[offset + j] - means[j]) * scales[j]
            j += 1


def sigmoid(z: Float64) -> Float64:
    if z >= 0.0:
        return 1.0 / (1.0 + exp(-z))
    var ez = exp(z)
    return ez / (1.0 + ez)


def clipped(v: Float64, limit: Float64) -> Float64:
    if v < -limit:
        return -limit
    if v > limit:
        return limit
    return v


def pa_tau(norm2: Float64, loss: Float64, c: Float64, mode: Int) -> Float64:
    if mode == 0:
        return loss / norm2 if norm2 > 0.0 else 0.0
    if mode == 1:
        var unconstrained = loss / norm2 if norm2 > 0.0 else 0.0
        return min(c, unconstrained)
    return loss / (norm2 + 0.5 / c)


def optimizer_step(
    weights: Ptr,
    grad: Ptr,
    state1: Ptr,
    state2: Ptr,
    d: Int,
    kind: Int,
    lr: Float64,
    beta1: Float64,
    beta2: Float64,
    eps: Float64,
):
    for j in range(d):
        var g = grad[j]
        if kind == 0:
            weights[j] -= lr * g
        elif kind == 1:
            state1[j] += g * g
            weights[j] -= lr * g / sqrt(state1[j] + eps)
        elif kind == 2:
            state1[j] = beta1 * state1[j] + (1.0 - beta1) * g * g
            weights[j] -= lr * g / sqrt(state1[j] + eps)
        else:
            state1[j] = beta1 * state1[j] + (1.0 - beta1) * g
            state2[j] = beta2 * state2[j] + (1.0 - beta2) * g * g
            weights[j] -= lr * state1[j] / (sqrt(state2[j]) + eps)


def learn_one(
    x: Ptr,
    weights: Ptr,
    grad: Ptr,
    state1: Ptr,
    state2: Ptr,
    d: Int,
    y: Float64,
    sample_weight: Float64,
    intercept: Float64,
    intercept_lr: Float64,
    l2: Float64,
    clip_gradient: Float64,
    kind: Int,
    lr: Float64,
    beta1: Float64,
    beta2: Float64,
    eps: Float64,
    logistic: Bool,
) -> Float64:
    var raw = dot(x, weights, d) + intercept
    var loss_grad = sigmoid(raw) - y if logistic else 2.0 * (raw - y)
    loss_grad = clipped(loss_grad * sample_weight, clip_gradient)
    for j in range(d):
        grad[j] = loss_grad * x[j] + l2 * weights[j]
    optimizer_step(weights, grad, state1, state2, d, kind, lr, beta1, beta2, eps)
    return intercept - intercept_lr * loss_grad


def learn_many(
    x: Ptr,
    y: Ptr,
    sample_weights: Ptr,
    weights: Ptr,
    grad: Ptr,
    state1: Ptr,
    state2: Ptr,
    n: Int,
    d: Int,
    intercept: Float64,
    intercept_lr: Float64,
    l2: Float64,
    clip_gradient: Float64,
    kind: Int,
    lr: Float64,
    beta1: Float64,
    beta2: Float64,
    eps: Float64,
    logistic: Bool,
) -> Float64:
    for j in range(d):
        grad[j] = 0.0
    var intercept_grad = 0.0
    for i in range(n):
        var raw = dot(x + i * d, weights, d) + intercept
        var loss_grad = sigmoid(raw) - y[i] if logistic else 2.0 * (raw - y[i])
        loss_grad = clipped(loss_grad * sample_weights[i], clip_gradient)
        intercept_grad += loss_grad
        for j in range(d):
            grad[j] += loss_grad * x[i * d + j]
    var inv_n = 1.0 / Float64(n)
    intercept_grad *= inv_n
    for j in range(d):
        grad[j] = grad[j] * inv_n + l2 * weights[j]
    optimizer_step(weights, grad, state1, state2, d, kind, lr, beta1, beta2, eps)
    return intercept - intercept_lr * intercept_grad


@export("mr_linear_learn_one")
def mr_linear_learn_one(
    x: Int, weights: Int, grad: Int, state1: Int, state2: Int, d: Int,
    y: Float64, sample_weight: Float64, intercept: Float64,
    intercept_lr: Float64, l2: Float64, clip_gradient: Float64,
    kind: Int, lr: Float64, beta1: Float64, beta2: Float64, eps: Float64,
) abi("C") -> Float64:
    return learn_one(
        p(x), p(weights), p(grad), p(state1), p(state2), d, y,
        sample_weight, intercept, intercept_lr, l2, clip_gradient,
        kind, lr, beta1, beta2, eps, False,
    )


@export("mr_logistic_learn_one")
def mr_logistic_learn_one(
    x: Int, weights: Int, grad: Int, state1: Int, state2: Int, d: Int,
    y: Float64, sample_weight: Float64, intercept: Float64,
    intercept_lr: Float64, l2: Float64, clip_gradient: Float64,
    kind: Int, lr: Float64, beta1: Float64, beta2: Float64, eps: Float64,
) abi("C") -> Float64:
    return learn_one(
        p(x), p(weights), p(grad), p(state1), p(state2), d, y,
        sample_weight, intercept, intercept_lr, l2, clip_gradient,
        kind, lr, beta1, beta2, eps, True,
    )


@export("mr_linear_learn_many")
def mr_linear_learn_many(
    x: Int, y: Int, sample_weights: Int, weights: Int, grad: Int,
    state1: Int, state2: Int, n: Int, d: Int, intercept: Float64,
    intercept_lr: Float64, l2: Float64, clip_gradient: Float64,
    kind: Int, lr: Float64, beta1: Float64, beta2: Float64, eps: Float64,
) abi("C") -> Float64:
    return learn_many(
        p(x), p(y), p(sample_weights), p(weights), p(grad), p(state1), p(state2),
        n, d, intercept, intercept_lr, l2, clip_gradient, kind, lr,
        beta1, beta2, eps, False,
    )


@export("mr_logistic_learn_many")
def mr_logistic_learn_many(
    x: Int, y: Int, sample_weights: Int, weights: Int, grad: Int,
    state1: Int, state2: Int, n: Int, d: Int, intercept: Float64,
    intercept_lr: Float64, l2: Float64, clip_gradient: Float64,
    kind: Int, lr: Float64, beta1: Float64, beta2: Float64, eps: Float64,
) abi("C") -> Float64:
    return learn_many(
        p(x), p(y), p(sample_weights), p(weights), p(grad), p(state1), p(state2),
        n, d, intercept, intercept_lr, l2, clip_gradient, kind, lr,
        beta1, beta2, eps, True,
    )


@export("mr_predict")
def mr_predict(
    x: Int, weights: Int, dst: Int, n: Int, d: Int, intercept: Float64,
    logistic: Int,
) abi("C"):
    var xp = p(x)
    var wp = p(weights)
    var dp = p(dst)
    for i in range(n):
        var raw = dot(xp + i * d, wp, d) + intercept
        dp[i] = sigmoid(raw) if logistic != 0 else raw


@export("mr_optimizer_step")
def mr_optimizer_step(
    weights: Int, grad: Int, state1: Int, state2: Int, d: Int, kind: Int,
    lr: Float64, beta1: Float64, beta2: Float64, eps: Float64,
) abi("C"):
    optimizer_step(
        p(weights), p(grad), p(state1), p(state2), d, kind, lr, beta1, beta2, eps
    )


@export("mr_pa_regression_learn")
def mr_pa_regression_learn(
    x: Int, weights: Int, d: Int, y: Float64, intercept: Float64,
    c: Float64, mode: Int, epsilon: Float64, learn_intercept: Int,
) abi("C") -> Float64:
    var xp = p(x)
    var wp = p(weights)
    var pred = dot(xp, wp, d) + intercept
    var error = y - pred
    var loss = abs(pred - (2.0 * y - 1.0)) - epsilon
    if loss <= 0.0:
        return intercept
    var norm2 = dot(xp, xp, d)
    var tau = pa_tau(norm2, loss, c, mode)
    var step = tau if error > 0.0 else -tau
    for j in range(d):
        wp[j] += step * xp[j]
    return intercept + step if learn_intercept != 0 else intercept


@export("mr_pa_classification_learn")
def mr_pa_classification_learn(
    x: Int, weights: Int, d: Int, y: Int, intercept: Float64,
    c: Float64, mode: Int, learn_intercept: Int,
) abi("C") -> Float64:
    var xp = p(x)
    var wp = p(weights)
    var raw = dot(xp, wp, d) + intercept
    var signed_y = 1.0 if y != 0 else -1.0
    var loss = max(0.0, 1.0 - signed_y * raw)
    var norm2 = dot(xp, xp, d)
    var tau = pa_tau(norm2, loss, c, mode)
    var step = tau * signed_y
    for j in range(d):
        wp[j] += step * xp[j]
    return intercept + step if learn_intercept != 0 else intercept


@export("mr_standard_update")
def mr_standard_update(
    x: Int, counts: Int, means: Int, variances: Int, n: Int, d: Int,
    with_std: Int,
) abi("C"):
    var xp = p(x)
    var cp = p(counts)
    var mp = p(means)
    var vp = p(variances)
    for i in range(n):
        for j in range(d):
            var value = xp[i * d + j]
            if value != value:
                continue
            cp[j] += 1.0
            var old_mean = mp[j]
            mp[j] += (value - old_mean) / cp[j]
            if with_std != 0:
                vp[j] += ((value - old_mean) * (value - mp[j]) - vp[j]) / cp[j]


@export("mr_standard_update_many")
def mr_standard_update_many(
    x: Int, counts: Int, means: Int, variances: Int, n: Int, d: Int,
    with_std: Int,
) abi("C"):
    comptime W = simdwidthof[DType.float64]()
    var xp = p(x)
    var cp = p(counts)
    var mp = p(means)
    var vp = p(variances)
    var vector_groups = d // W
    var use_std = with_std != 0
    for group in range(vector_groups):
        standard_update_vector(
            xp, cp, mp, vp, n, d, group * W, use_std
        )
    for j in range(vector_groups * W, d):
        standard_update_scalar(xp, cp, mp, vp, n, d, j, use_std)


@export("mr_standard_transform")
def mr_standard_transform(
    x: Int, means: Int, variances: Int, dst: Int, n: Int, d: Int,
    with_std: Int,
) abi("C"):
    var xp = p(x)
    var mp = p(means)
    var vp = p(variances)
    var dp = p(dst)
    for j in range(d):
        vp[j] = 1.0 / sqrt(vp[j]) if with_std != 0 and vp[j] > 0.0 else 1.0
    standard_transform_rows(xp, mp, vp, dp, d, 0, n)


@export("mr_normalize")
def mr_normalize(x: Int, dst: Int, n: Int, d: Int) abi("C"):
    var xp = p(x)
    var dp = p(dst)
    for i in range(n):
        var norm = sqrt(dot(xp + i * d, xp + i * d, d))
        var inv = 1.0 / norm if norm > 0.0 else 0.0
        for j in range(d):
            dp[i * d + j] = xp[i * d + j] * inv


@export("mr_sum_absolute_error")
def mr_sum_absolute_error(y: Int, pred: Int, weights: Int, n: Int) abi("C") -> Float64:
    var yp = p(y)
    var pp = p(pred)
    var wp = p(weights)
    var total = 0.0
    for i in range(n):
        total += wp[i] * abs(yp[i] - pp[i])
    return total


@export("mr_sum_squared_error")
def mr_sum_squared_error(y: Int, pred: Int, weights: Int, n: Int) abi("C") -> Float64:
    var yp = p(y)
    var pp = p(pred)
    var wp = p(weights)
    var total = 0.0
    for i in range(n):
        var error = yp[i] - pp[i]
        total += wp[i] * error * error
    return total


@export("mr_sum_log_loss")
def mr_sum_log_loss(y: Int, pred: Int, weights: Int, n: Int) abi("C") -> Float64:
    var yp = p(y)
    var pp = p(pred)
    var wp = p(weights)
    var total = 0.0
    for i in range(n):
        var prob = max(1e-15, min(1.0 - 1e-15, pp[i]))
        total -= wp[i] * (yp[i] * log(prob) + (1.0 - yp[i]) * log(1.0 - prob))
    return total


@export("mr_sum_accuracy")
def mr_sum_accuracy(y: Int, pred: Int, weights: Int, n: Int) abi("C") -> Float64:
    var yp = p(y)
    var pp = p(pred)
    var wp = p(weights)
    var total = 0.0
    for i in range(n):
        if yp[i] == pp[i]:
            total += wp[i]
    return total
