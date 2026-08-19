"""Functions for evaluating the predictions of the OpenADMET CYP blind challenge.

Ported from the challenge backend, so it should match exactly what you see on the leaderboard!
"""

import inspect
from typing import Callable

import numpy as np
import pandas as pd
from loguru import logger

from .config import (
    ACTIVITY_METRICS,
    BOOTSTRAP_SAMPLES,
    CLASSIFICATION_ENDPOINTS,
    CLASSIFICATION_METRICS,
    ENDPOINTS_TO_LOG_TRANSFORM,
    MACRO_ENDPOINT_LABEL,
    METRIC_NAN_FALLBACK,
    REGRESSION_CREDIBLE_INTERVALS_LOWER_SUFFIX,
    REGRESSION_CREDIBLE_INTERVALS_UPPER_SUFFIX,
)
from .utils import bootstrap_sampling, clip_and_log_transform


# ---------------------------------------------------------------------------
# Activity scoring
# ---------------------------------------------------------------------------


def _metrics_for_endpoint(endpoint: str) -> list[tuple[str, Callable]]:
    """Return the metric list to use for a given activity endpoint.

    Classification (TDI) endpoints are scored with ``CLASSIFICATION_METRICS``
    (MCC/Accuracy/Precision/Recall/F1); every other activity endpoint (regression,
    direct-inhibition pIC50) is scored with ``ACTIVITY_METRICS``.
    """
    return CLASSIFICATION_METRICS if endpoint in CLASSIFICATION_ENDPOINTS else ACTIVITY_METRICS


def score_activity_predictions(
    predictions: pd.DataFrame, ground_truth: pd.DataFrame, endpoints: list[str]
) -> pd.DataFrame:
    """Score the activity predictions against the ground truth.

    Metrics are calculated for bootstrapped samples of the dataset to allow for testing
    the statistical significance of differences between submissions. Each endpoint is
    scored with the metric list appropriate to its type — regression endpoints get
    ``ACTIVITY_METRICS``, classification (TDI) endpoints get ``CLASSIFICATION_METRICS``
    — see ``_metrics_for_endpoint``.

    Each endpoint is scored only on the compounds that have a ground-truth value for
    that endpoint — a compound not tested for a given endpoint has ``y_true == NaN``
    there and is excluded from that endpoint's bootstrap sampling entirely. Every
    compound with a ground-truth value is expected to have a prediction (participants
    are asked to predict every compound, and submission_validation.py is the primary
    check for missing predictions); a NaN prediction for such a compound is treated as
    a validation failure here too.

    Regression endpoints carry credible-interval bound columns in ``ground_truth``
    (named ``f"{endpoint}{REGRESSION_CREDIBLE_INTERVALS_UPPER_SUFFIX}"`` /
    ``f"{endpoint}{REGRESSION_CREDIBLE_INTERVALS_LOWER_SUFFIX}"``), used by the
    soft-thresholded RAE metric (``ST-RAE``). When present, these are threaded through
    to ``bootstrap_metrics`` alongside ``y_true``/``y_pred``; classification endpoints
    have no such columns, so ``None`` is passed instead (harmless, since none of
    ``CLASSIFICATION_METRICS`` consume them).

    This function does not compute the macro-averaged "MA" pseudo-endpoint. Callers
    that want a track's "MA" row should call ``add_macro_endpoint`` on this function's
    output with the same ``endpoints`` (and that track's own metrics).

    Args:
        predictions (pd.DataFrame): The predicted activity values.
        ground_truth (pd.DataFrame): The true activity values.
        endpoints (list[str]): The endpoints to score, e.g. ``REGRESSION_ENDPOINTS``
            or ``CLASSIFICATION_ENDPOINTS`` — regression and classification are
            independent submission tracks, so a given call only ever scores one
            track's endpoints.

    Returns:
        pd.DataFrame: A DataFrame containing the scored bootstrapped activity
            predictions, one row per (endpoint, bootstrap sample) — no macro
            pseudo-endpoint included.

    Raises:
        ValueError: If a compound with a ground-truth value for an endpoint has no
            prediction for that endpoint.

    """
    logger.info("Scoring activity predictions against ground truth")
    merged_df = predictions.merge(
        ground_truth, on="Molecule_Name", suffixes=("_pred", "_true"), how="right"
    ).sort_values("Molecule_Name")
    logger.info(
        "Completed merging predictions with ground truth. Merged dataset contains {} "
        "rows and {} columns.",
        merged_df.shape[0],
        merged_df.shape[1],
    )

    all_endpoint_bootstrap_results_list = []
    for endpoint in endpoints:
        logger.info("Scoring endpoint: {}", endpoint)
        y_pred = merged_df[f"{endpoint}_pred"].to_numpy()
        y_true = merged_df[f"{endpoint}_true"].to_numpy()

        # Credible-interval bound columns aren't merge-suffixed: they only ever come
        # from ground_truth (predictions never carry them), so they keep their plain
        # names — see merge() above (suffixes only apply to overlapping columns).
        upper_col = f"{endpoint}{REGRESSION_CREDIBLE_INTERVALS_UPPER_SUFFIX}"
        lower_col = f"{endpoint}{REGRESSION_CREDIBLE_INTERVALS_LOWER_SUFFIX}"
        y_true_upper = (
            merged_df[upper_col].to_numpy() if upper_col in merged_df.columns else None
        )
        y_true_lower = (
            merged_df[lower_col].to_numpy() if lower_col in merged_df.columns else None
        )

        # pd.isna (not np.isnan) so this works for classification endpoints too —
        # their ground-truth column can be object/bool dtype (e.g. CYP2D6_is_TDI has
        # a couple of genuinely missing labels), which np.isnan can't handle.
        has_ground_truth = ~pd.isna(y_true)
        if not has_ground_truth.all():
            logger.debug(
                "Excluding {} compound(s) with no ground truth for endpoint {}",
                (~has_ground_truth).sum(),
                endpoint,
            )
            y_pred = y_pred[has_ground_truth]
            y_true = y_true[has_ground_truth]
            if y_true_upper is not None:
                y_true_upper = y_true_upper[has_ground_truth]
            if y_true_lower is not None:
                y_true_lower = y_true_lower[has_ground_truth]

        # A submission itself must never contain NaN (submission_validation.py's
        # nullable=False already rejects that) — this instead defends against a
        # prediction going missing specifically for a compound that *does* have
        # ground truth, which validation of the raw submission can't catch on its own.
        if pd.isna(y_pred).any():
            raise ValueError(
                f"Missing prediction(s) for endpoint '{endpoint}': every compound "
                "with a ground-truth value must have a prediction."
            )

        if endpoint in CLASSIFICATION_ENDPOINTS:
            # Safe only after NaN rows have already been dropped from both arrays.
            y_true = y_true.astype(bool)
            y_pred = y_pred.astype(bool)
        elif endpoint in ENDPOINTS_TO_LOG_TRANSFORM:
            logger.debug("Applying log transformation to endpoint {}", endpoint)
            y_pred = clip_and_log_transform(y_pred)
            y_true = clip_and_log_transform(y_true)

        bootstrap_df = bootstrap_metrics(
            y_pred,
            y_true,
            endpoint,
            n_bootstrap_samples=BOOTSTRAP_SAMPLES,
            metrics=_metrics_for_endpoint(endpoint),
            y_true_upper=y_true_upper,
            y_true_lower=y_true_lower,
        )
        all_endpoint_bootstrap_results_list.append(bootstrap_df)
    all_endpoint_bootstrap_results = pd.concat(
        all_endpoint_bootstrap_results_list, ignore_index=True
    )
    logger.info("Completed scoring activity predictions")
    return all_endpoint_bootstrap_results


def add_macro_endpoint(
    all_endpoint_bootstrap_results: pd.DataFrame,
    endpoints: list[str],
    metrics: list[tuple[str, Callable]],
) -> pd.DataFrame:
    """Narrow to one track's endpoints/metrics and append its macro "MA" row.

    ``score_activity_predictions`` scores every activity endpoint (regression and
    classification) in one call, concatenating per-endpoint frames that have
    *different* metric columns (regression rows have MAE/ST-RAE/..., classification rows
    have MCC/Accuracy/...) — the concatenated result has both sets of columns, NaN
    wherever a metric doesn't apply to that row's endpoint. This filters rows down to
    just ``endpoints`` (one track's real endpoints) and columns down to just
    ``metrics`` (that track's own metrics), so no cross-track NaN columns leak into
    the result, then appends a macro-averaged "MA" row set (via
    ``compute_macro_bootstrap_results``) when there's more than one endpoint to
    average across.

    Args:
        all_endpoint_bootstrap_results (pd.DataFrame): Output of
            ``score_activity_predictions`` (or any frame with "Sample", "Endpoint",
            and metric columns for multiple endpoints/tracks).
        endpoints (list[str]): The track's real endpoints to keep, e.g.
            ``REGRESSION_ENDPOINTS`` or ``CLASSIFICATION_ENDPOINTS``. May be empty, in
            which case the result is empty (callers should generally avoid calling
            this with an empty list rather than relying on that).
        metrics (list[tuple[str, Callable]]): The track's own metric list, e.g.
            ``ACTIVITY_METRICS`` or ``CLASSIFICATION_METRICS`` — only these columns
            are kept.

    Returns:
        pd.DataFrame: This track's real-endpoint rows, narrowed to its own metric
            columns, plus a macro "MA" row set when ``len(endpoints) > 1``.

    """
    metric_names = [name for name, _ in metrics]
    track_results = all_endpoint_bootstrap_results[
        all_endpoint_bootstrap_results["Endpoint"].isin(endpoints)
    ][["Sample", "Endpoint", *metric_names]]

    if len(endpoints) > 1:
        logger.info(
            "Calculating macro-averaged metrics across endpoints for each bootstrap sample"
        )
        macro_bootstrap_results = compute_macro_bootstrap_results(
            track_results, metrics=metrics
        )
        track_results = pd.concat(
            [track_results, macro_bootstrap_results], ignore_index=True
        )
    return track_results


def compute_macro_bootstrap_results(
    all_endpoint_bootstrap_results: pd.DataFrame,
    metrics: list[tuple[str, Callable]],
) -> pd.DataFrame:
    """Compute per-bootstrap-sample macro-averaged metrics across all endpoints.

    For every bootstrap sample, every metric in ``metrics`` is macro-averaged across
    endpoints with a plain arithmetic mean.

    Spearman_R was previously averaged via a Fisher z-transform (``arctanh`` /
    ``tanh``), the standard variance-stabilising treatment for combining several
    noisy *estimates of the same underlying correlation* (e.g. meta-analysis, or
    averaging one endpoint's Spearman across repeated resamples of the same data).
    That doesn't apply here: this average combines Spearman scores from *different*
    endpoints (different isoforms), which are unrelated true correlations, not
    repeated estimates of one. Fisher's z blows up near +/-1 (``arctanh(1) = inf``,
    clipped in practice but still huge — e.g. ``arctanh(1 - 1e-7) ≈ 8.4`` vs.
    ``arctanh(0) = 0``), so a submission with a near-perfect Spearman on a couple of
    endpoints and ~0 on the rest could macro-average to ~0.99 instead of the
    naively-expected ~0.5, letting a handful of easy/lucky endpoints dominate the
    macro score. A plain mean — already used for ST-RAE/MAE/R2/Kendall_Tau — doesn't have
    this failure mode, so Spearman_R now uses one too, for the same reason
    Kendall_Tau always has: Fisher's z-transform has no standard extension to
    Kendall's tau (different asymptotic sampling distribution), so there was never a
    transform-based option for it here.

    Args:
        all_endpoint_bootstrap_results (pd.DataFrame): Per-endpoint bootstrap metrics
            for a single track, as returned by ``add_macro_endpoint``'s narrowing step
            (or by concatenating per-endpoint ``bootstrap_metrics(...)`` results).
            Must contain "Sample", "Endpoint", and one column per metric in
            ``metrics``.
        metrics (list[tuple[str, Callable]]): The metric list to macro-average — only
            the names are used here (e.g. ``ACTIVITY_METRICS`` or
            ``CLASSIFICATION_METRICS``).

    Returns:
        pd.DataFrame: One row per bootstrap sample, with columns "Sample",
            "Endpoint" (``MACRO_ENDPOINT_LABEL`` for every row), and the macro-averaged
            value of each metric in ``metrics`` for that sample.

    """
    grouped = all_endpoint_bootstrap_results.groupby("Sample")
    macro_results = pd.DataFrame(index=grouped.size().index)
    for metric_name, _ in metrics:
        logger.info(
            "Computing macro-averaged metric {} across bootstrap iterations",
            metric_name,
        )
        macro_results[metric_name] = grouped[metric_name].mean()
    macro_results = macro_results.reset_index()
    macro_results["Endpoint"] = MACRO_ENDPOINT_LABEL
    return macro_results


def pivot_endpoint_results_wide(by_endpoint_results: pd.DataFrame) -> pd.DataFrame:
    """Pivot per-endpoint mean/std results into a single wide row.

    ``by_endpoint_results`` (as returned by ``average_bootstrap_results_by_endpoint``)
    has one row per endpoint (indexed by endpoint name, including the synthetic
    ``MACRO_ENDPOINT_LABEL`` ("MA") pseudo-endpoint computed by
    ``compute_macro_bootstrap_results``) and one column per ``<metric>_mean`` /
    ``<metric>_std``. This flattens it into a single-row DataFrame suitable for saving
    as a submission's ``averaged-results.parquet``, with every endpoint's columns
    consistently prefixed as ``f"{endpoint}_{metric}_{mean|std}"`` (e.g.
    ``"CYP3A4_pIC50_active_MAE_mean"``, ``"MA_ST-RAE_mean"``).

    A leaderboard for any single endpoint (macro or real) is then built by narrowing
    back down to that endpoint's columns and stripping the prefix, so ``primary_metric``
    is always a bare metric name (e.g. ``"ST-RAE"``) regardless of which endpoint a
    given leaderboard targets.

    Args:
        by_endpoint_results (pd.DataFrame): Per-endpoint mean/std results, indexed by
            endpoint name (including ``MACRO_ENDPOINT_LABEL``).

    Returns:
        pd.DataFrame: A single-row DataFrame with one column per
            endpoint/metric/statistic combination.

    """
    wide_row: dict[str, float] = {}
    for endpoint, row in by_endpoint_results.iterrows():
        for column, value in row.items():
            wide_row[f"{endpoint}_{column}"] = value
    return pd.DataFrame([wide_row])


def average_bootstrap_results_by_endpoint(
    all_endpoint_bootstrap_results: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate the average results of the bootstrapped samples for each endpoint.

    Args:
        all_endpoint_bootstrap_results (pd.DataFrame): A DataFrame containing the
            bootstrapped results for each endpoint.

    Returns:
        pd.DataFrame: A DataFrame containing the average results of the bootstrapped
                      samples.

    """
    logger.info("Calculating average bootstrap results by endpoint")
    agg_df = (
        all_endpoint_bootstrap_results.set_index("Sample")
        .groupby("Endpoint")
        .agg(["mean", "std"])
    )
    agg_df.columns = ["_".join(col).strip() for col in agg_df.columns.values]
    return agg_df


def _metric_needs_credible_interval_bounds(metric_func: Callable) -> bool:
    """True if ``metric_func`` accepts ``y_true_upper``/``y_true_lower`` keywords.

    Lets ``bootstrap_metrics`` dispatch the credible-interval bounds only to metrics
    that use them (e.g. ``rae_soft_threshold_absolute_error``), while other metrics in
    the same list (MAE, R2, ...) keep their plain two-argument call — introspecting
    the signature avoids hardcoding metric names here.
    """
    params = inspect.signature(metric_func).parameters
    return "y_true_upper" in params and "y_true_lower" in params


def bootstrap_metrics(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    endpoint: str,
    n_bootstrap_samples: int,
    metrics: list[tuple[str, Callable]] = ACTIVITY_METRICS,
    y_true_upper: np.ndarray | None = None,
    y_true_lower: np.ndarray | None = None,
) -> pd.DataFrame:
    """Calculate bootstrap metrics given predicted and true values.

    Args:
        y_pred (np.ndarray): The predicted values.
        y_true (np.ndarray): The true values.
        endpoint (str): The endpoint for which the metrics are being calculated.
        n_bootstrap_samples (int): The number of bootstrap samples to generate.
        metrics (list[tuple[str, Callable]]): The ``(name, func)`` metric list to
            compute for every bootstrap sample — ``ACTIVITY_METRICS`` for a
            regression endpoint, ``CLASSIFICATION_METRICS`` for a classification
            (TDI) endpoint. Defaults to ``ACTIVITY_METRICS``.
        y_true_upper (np.ndarray | None): Per-compound upper credible-interval bound
            for ``y_true``, aligned with ``y_true``/``y_pred``. Only consumed by
            metrics whose signature accepts ``y_true_upper``/``y_true_lower`` (see
            ``_metric_needs_credible_interval_bounds``), e.g. the soft-thresholded
            RAE metric — ignored by every other metric. Required if ``metrics``
            includes such a metric, otherwise optional.
        y_true_lower (np.ndarray | None): Per-compound lower credible-interval bound,
            counterpart to ``y_true_upper``.

    Returns:
        pd.DataFrame: A DataFrame containing the bootstrap metrics for the given
                      endpoint.

    Raises:
        RuntimeError: If a metric cannot be calculated, or returns a non-finite
            value with no entry in ``METRIC_NAN_FALLBACK``, for any bootstrap sample
            — rather than silently scoring that sample as 0 (which would misrepresent
            a real failure as a perfect score for error metrics like MAE/ST-RAE).
            Metrics listed in ``METRIC_NAN_FALLBACK`` (e.g. Spearman_R/Kendall_Tau,
            which are mathematically undefined for a zero-variance bootstrap sample —
            such as a submission predicting the same value for every compound) use
            that fallback value instead of raising. This also covers a metric that
            needs credible-interval bounds (e.g. ST-RAE) when none were supplied.

    """
    metrics_with_bounds_flag = [
        (name, func, _metric_needs_credible_interval_bounds(func))
        for name, func in metrics
    ]

    bootstrap_metrics_list = []
    for bootstrap_iteration, idx in enumerate(
        bootstrap_sampling(y_true.shape[0], n_bootstrap_samples)
    ):
        metric_values = {"Sample": bootstrap_iteration, "Endpoint": endpoint}
        for metric_name, metric_func, needs_bounds in metrics_with_bounds_flag:
            try:
                if needs_bounds:
                    if y_true_upper is None or y_true_lower is None:
                        raise ValueError(
                            f"Metric '{metric_name}' requires credible-interval "
                            "bounds (y_true_upper/y_true_lower), but none were "
                            "provided to bootstrap_metrics."
                        )
                    metric_value = metric_func(
                        y_true[idx],
                        y_pred[idx],
                        y_true_upper=y_true_upper[idx],
                        y_true_lower=y_true_lower[idx],
                    )
                else:
                    metric_value = metric_func(y_true[idx], y_pred[idx])
                if not isinstance(metric_value, (int, float)):
                    metric_value = metric_value.statistic
            except Exception as e:
                raise RuntimeError(
                    f"Error calculating metric '{metric_name}' for endpoint "
                    f"'{endpoint}' (bootstrap sample {bootstrap_iteration}): {e}"
                ) from e
            if not np.isfinite(metric_value):
                if metric_name not in METRIC_NAN_FALLBACK:
                    raise RuntimeError(
                        f"Metric '{metric_name}' for endpoint '{endpoint}' "
                        f"(bootstrap sample {bootstrap_iteration}) returned a "
                        f"non-finite value: {metric_value}"
                    )
                metric_value = METRIC_NAN_FALLBACK[metric_name]
            metric_values[metric_name] = metric_value
        bootstrap_metrics_list.append(metric_values)

    bootstrap_df = pd.DataFrame(bootstrap_metrics_list)
    return bootstrap_df
