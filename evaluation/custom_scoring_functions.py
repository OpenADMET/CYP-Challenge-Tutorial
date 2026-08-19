"""Custom scoring functions for regression tasks."""

import numpy as np
import pandas as pd

# Minimum confidence-interval width used in rae_inverse_confidence_weighting, to stop
# a near-zero interval from producing an arbitrarily large (or infinite) weight.
_MIN_CONFIDENCE_INTERVAL = 1e-6


def rae(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> float:
    """Relative absolute error (RAE) metric for regression tasks.

    Args:
        y_true (pd.Series | np.ndarray): True values.
        y_pred (pd.Series | np.ndarray): Predicted values.

    Returns:
        float: The relative absolute error (RAE) score.

    """
    return np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true - np.mean(y_true)))


def _resolve_bounds(
    y_true: pd.Series | np.ndarray,
    y_true_upper: pd.Series | np.ndarray | None,
    y_true_lower: pd.Series | np.ndarray | None,
    confidence_interval: float | pd.Series | np.ndarray | None,
) -> tuple[pd.Series | np.ndarray, pd.Series | np.ndarray]:
    """Resolve upper/lower bounds from either explicit bounds or a confidence interval.

    ``confidence_interval`` and explicit bounds (``y_true_upper``/``y_true_lower``) are
    mutually exclusive. ``confidence_interval`` is treated as the full width of a band
    centred on ``y_true`` (i.e. ``y_true +/- confidence_interval / 2``).

    A side of the band that is neither given explicitly nor derivable from
    ``confidence_interval`` defaults to ``y_true`` itself — i.e. no tolerance on that
    side, since there's no information to define one. If both sides default this way
    (no bounds and no confidence_interval given at all), the band collapses to the
    point estimate on both sides, which — for callers built on top of this, like
    ``rae_soft_threshold_absolute_error`` — makes soft-thresholding a no-op and
    recovers the plain (non-thresholded) behaviour exactly.

    Args:
        y_true (pd.Series | np.ndarray): True values.
        y_true_upper (pd.Series | np.ndarray | None): Optional upper bounds for true
            values.
        y_true_lower (pd.Series | np.ndarray | None): Optional lower bounds for true
            values.
        confidence_interval (float | pd.Series | np.ndarray | None): Optional
            confidence interval (full width) for true values.

    Returns:
        tuple[pd.Series | np.ndarray, pd.Series | np.ndarray]: The resolved
            ``(y_true_lower, y_true_upper)`` bounds.

    Raises:
        ValueError: If both explicit bounds and a confidence interval are provided.

    """
    if (
        y_true_upper is not None or y_true_lower is not None
    ) and confidence_interval is not None:
        raise ValueError(
            "Cannot provide both upper/lower bounds and confidence interval for soft thresholding."
        )

    if confidence_interval is not None:
        half_width = confidence_interval / 2
        y_true_upper = y_true + half_width
        y_true_lower = y_true - half_width
    else:
        if y_true_upper is None:
            y_true_upper = y_true
        if y_true_lower is None:
            y_true_lower = y_true

    return y_true_lower, y_true_upper


def rae_soft_threshold_absolute_error(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    y_true_upper: pd.Series | np.ndarray | None = None,
    y_true_lower: pd.Series | np.ndarray | None = None,
    confidence_interval: float | pd.Series | np.ndarray | None = None,
) -> float:
    """RAE metric for regression tasks, with soft thresholding.

    If a confidence interval is provided, the absolute error is clipped to the distance
    to the nearest bound defined by the confidence interval. The confidence interval can
    be a single float or an array of the same shape as y_true/y_pred. If upper/lower
    bounds are provided instead, the absolute error is clipped to the distance to the
    nearest bound. Explicit bounds and a confidence interval are mutually exclusive.

    A prediction that falls inside the ``[y_true_lower, y_true_upper]`` tolerance band
    contributes zero error — it is treated as indistinguishable from the true value
    given measurement uncertainty. A prediction outside the band contributes only the
    distance to the nearest edge, rather than the distance to the point estimate.

    Either ``y_true_upper`` or ``y_true_lower`` may be omitted on its own — that side
    then defaults to ``y_true`` itself (no tolerance on that side; see
    ``_resolve_bounds``). Omitting both, and no confidence_interval either, collapses
    the band to a single point on both sides, which makes soft-thresholding a no-op:
    the result is then identical to plain ``rae()``.

    The naive baseline in the denominator (a constant predictor at ``mean(y_true)``) is
    put through the same soft-thresholding as the model's predictions, so both halves
    of the ratio are computed under the same rule and RAE keeps its usual meaning:
    ``1.0`` means the model is exactly as good as always predicting the mean, under
    this tolerance-band error function.

    Args:
        y_true (pd.Series | np.ndarray): True values.
        y_pred (pd.Series | np.ndarray): Predicted values.
        y_true_upper (pd.Series | np.ndarray | None): Optional upper bounds for true
            values. Defaults to ``y_true`` (no tolerance above) if omitted.
        y_true_lower (pd.Series | np.ndarray | None): Optional lower bounds for true
            values. Defaults to ``y_true`` (no tolerance below) if omitted.
        confidence_interval (float | pd.Series | np.ndarray | None): Optional
            confidence interval (full width) for true values.

    Returns:
        float: The relative absolute error (RAE) score, with soft-thresholded
            absolute error in both the model error and the naive-baseline error.

    Raises:
        ValueError: If both explicit bounds and a confidence interval are provided.

    """
    y_true_lower, y_true_upper = _resolve_bounds(
        y_true, y_true_upper, y_true_lower, confidence_interval
    )

    above_upper = np.clip(y_pred - y_true_upper, a_min=0, a_max=None)
    below_lower = np.clip(y_true_lower - y_pred, a_min=0, a_max=None)
    soft_abs_error = above_upper + below_lower

    mean_true = np.mean(y_true)
    baseline_above_upper = np.clip(mean_true - y_true_upper, a_min=0, a_max=None)
    baseline_below_lower = np.clip(y_true_lower - mean_true, a_min=0, a_max=None)
    soft_baseline_error = baseline_above_upper + baseline_below_lower

    return np.sum(soft_abs_error) / np.sum(soft_baseline_error)


def _weighted_absolute_error_below_threshold(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray | float,
    threshold: float,
    weighting: float,
) -> pd.Series | np.ndarray:
    """Per-point absolute error, downweighted where both y_true and y_pred are below threshold.

    Args:
        y_true (pd.Series | np.ndarray): True values.
        y_pred (pd.Series | np.ndarray | float): Predicted values, or a single constant
            prediction (e.g. ``mean(y_true)`` for a naive baseline) broadcast against
            ``y_true``.
        threshold (float): The threshold below which the absolute error is weighted.
        weighting (float): The factor by which to weight the absolute error below the
            threshold.

    Returns:
        pd.Series | np.ndarray: Per-point weighted absolute error.

    """
    abs_error = np.abs(y_true - y_pred)
    below_threshold = (y_true < threshold) & (y_pred < threshold)
    weights = np.where(below_threshold, weighting, 1.0)
    return weights * abs_error


def rae_weight_below_threshold(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    threshold: float = 4,
    weighting: float = 0.25,
) -> float:
    """RAE metric for regression tasks, weighted below a threshold.

    The absolute error is weighted by a factor if the ground truth and prediction are
    both below a certain threshold. Requiring both — rather than just the true value —
    to be below the threshold means a real, qualitative miss (e.g. true is below
    threshold but predicted well above it, or vice versa) is still scored at full
    weight; only errors confined to the low-confidence region are downweighted.

    The naive baseline in the denominator is treated as a constant predictor at
    ``mean(y_true)`` and put through this *exact same* weighting rule (i.e. its
    "prediction" for the below-threshold check is ``mean(y_true)`` itself, not
    ``y_true`` alone) — so both halves of the ratio are computed identically and RAE
    keeps its usual meaning: ``1.0`` means the model is exactly as good as always
    predicting the mean, under this weighting. In practice, ``mean(y_true)`` for a
    pIC50-like dataset is almost always above the threshold, so the baseline's
    below-threshold condition is rarely satisfied and the denominator ends up close to
    the unweighted RAE denominator — the weighting mostly changes the numerator.

    Args:
        y_true (pd.Series | np.ndarray): True values.
        y_pred (pd.Series | np.ndarray): Predicted values.
        threshold (float): The threshold below which the absolute error is weighted.
            Defaults to 4.
        weighting (float): The factor by which to weight the absolute error below the
            threshold. Defaults to 0.25.

    Returns:
        float: The relative absolute error (RAE) score, with weighted absolute error
            below the threshold in both the model error and the naive-baseline error.

    """
    mean_true = np.mean(y_true)

    weighted_error = _weighted_absolute_error_below_threshold(
        y_true, y_pred, threshold, weighting
    )
    weighted_baseline_error = _weighted_absolute_error_below_threshold(
        y_true, mean_true, threshold, weighting
    )

    return np.sum(weighted_error) / np.sum(weighted_baseline_error)


def rae_inverse_confidence_weighting(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    y_true_upper: pd.Series | np.ndarray | None = None,
    y_true_lower: pd.Series | np.ndarray | None = None,
    confidence_interval: float | pd.Series | np.ndarray | None = None,
) -> float:
    """RAE metric for regression tasks, with inverse confidence weighting.

    If a confidence interval is provided, the absolute error is weighted by the inverse
    of the confidence interval. The confidence interval can be a single float or an
    array of the same shape as y_true/y_pred. If upper/lower bounds are provided, the
    absolute error is weighted by the inverse of the range of the upper and lower bounds.
    If both or neither are provided, an error is raised.

    Unlike ``rae_soft_threshold_absolute_error``, this scales every point's
    contribution continuously by how (un)certain its true value is, rather than
    applying a hard tolerance band.

    The naive baseline in the denominator is the constant predictor that minimises the
    *weighted squared* error — i.e. the weighted mean ``sum(weights * y_true) /
    sum(weights)`` using these same per-point inverse-confidence weights — not the
    plain, unweighted ``mean(y_true)``. This is the direct weighted analogue of how
    plain RAE's baseline (the arithmetic mean) is used: it's the weighted
    least-squares-optimal constant, *not* the weighted-L1-optimal constant (that would
    be the weighted median) — plain RAE has this same quirk, using the mean rather
    than the true minimiser of its own (L1) error function, so this preserves that
    convention rather than fixing it. Since these weights depend only on ``y_true``'s
    own uncertainty and not on any prediction, the weighted mean has a closed form,
    unlike in ``rae_weight_below_threshold`` (where the weighting itself depends on the
    candidate prediction, making a closed-form weighted mean ill-defined, so that
    function keeps the plain mean instead). Both halves of the ratio are computed
    under the same rule, so predicting the weighted mean exactly gives ``RAE = 1.0``
    by construction — though, as with plain RAE, some other constant (e.g. the
    weighted median) can occasionally score lower than this baseline.

    A side of the bound left unspecified defaults to ``y_true`` itself (see
    ``_resolve_bounds``) — e.g. omitting both bounds and confidence_interval entirely
    collapses ``interval_width`` to ``0`` everywhere (clipped up to
    ``_MIN_CONFIDENCE_INTERVAL``), giving every point the same weight, which reduces
    this metric to plain ``rae()`` exactly (the constant weight cancels out of both
    the numerator and denominator).

    Args:
        y_true (pd.Series | np.ndarray): True values.
        y_pred (pd.Series | np.ndarray): Predicted values.
        y_true_upper (pd.Series | np.ndarray | None): Optional upper bounds for true
            values. Defaults to ``y_true`` if omitted.
        y_true_lower (pd.Series | np.ndarray | None): Optional lower bounds for true
            values. Defaults to ``y_true`` if omitted.
        confidence_interval (float | pd.Series | np.ndarray | None): Optional
            confidence interval (full width) for true values.

    Returns:
        float: The relative absolute error (RAE) score, with inverse confidence
            weighting applied to both the model error and the naive-baseline error.

    Raises:
        ValueError: If both explicit bounds and a confidence interval are provided.

    """
    y_true_lower, y_true_upper = _resolve_bounds(
        y_true, y_true_upper, y_true_lower, confidence_interval
    )
    interval_width = np.clip(
        y_true_upper - y_true_lower, a_min=_MIN_CONFIDENCE_INTERVAL, a_max=None
    )
    weights = 1.0 / interval_width

    abs_error = np.abs(y_true - y_pred)

    # Weighted-least-squares-optimal constant, i.e. the weighted analogue of
    # mean(y_true) — the best a naive constant predictor can do under this weighting.
    weighted_mean = np.sum(weights * y_true) / np.sum(weights)
    baseline_error = np.abs(y_true - weighted_mean)

    return np.sum(weights * abs_error) / np.sum(weights * baseline_error)
