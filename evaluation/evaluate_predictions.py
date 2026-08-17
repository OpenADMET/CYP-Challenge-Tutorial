"""Functions for evaluating the predictions of the OpenADMET CYP blind challenge."""

import numpy as np
import pandas as pd
from loguru import logger

from .config import (
    ACTIVITY_METRICS,
    BOOTSTRAP_SAMPLES,
    ENDPOINTS,
    ENDPOINTS_TO_LOG_TRANSFORM,
    TDI_ENDPOINT,
    TDI_METRICS,
)
from .utils import bootstrap_sampling, clip_and_log_transform


# ---------------------------------------------------------------------------
# Activity scoring
# ---------------------------------------------------------------------------


def score_activity_predictions(
    predictions: pd.DataFrame, ground_truth: pd.DataFrame
) -> pd.DataFrame:
    """Score the activity predictions against the ground truth.

    Metrics are calculated for bootstrapped samples of the dataset to allow for testing
    the statistical significance of differences between submissions.

    Args:
        predictions (pd.DataFrame): The predicted activity values.
        ground_truth (pd.DataFrame): The true activity values.

    Returns:
        pd.DataFrame: A DataFrame containing the scored bootstrapped activity
                      predictions.

    Raises:
        ValueError: If the merged DataFrame contains NaN values after merging
                    predictions with ground truth.

    """
    logger.info("Scoring activity predictions against ground truth")
    merged_df = predictions.merge(
        ground_truth, on="Molecule Name", suffixes=("_pred", "_true"), how="right"
    ).sort_values("Molecule Name")
    logger.info(
        "Completed merging predictions with ground truth. Merged dataset contains {} "
        "rows and {} columns.",
        merged_df.shape[0],
        merged_df.shape[1],
    )

    if merged_df.isnull().any().any():
        logger.warning(
            "Merged DataFrame contains NaN values after merging predictions with ground"
            " truth. This may indicate missing predictions for some molecules."
        )
        raise ValueError(
            "Merged DataFrame contains NaN values after merging predictions with ground truth."
        )

    all_endpoint_bootstrap_results_list = []
    for endpoint in ENDPOINTS:
        logger.info("Scoring endpoint: {}", endpoint)
        y_pred = merged_df[f"{endpoint}_pred"].to_numpy()
        y_true = merged_df[f"{endpoint}_true"].to_numpy()

        if endpoint in ENDPOINTS_TO_LOG_TRANSFORM:
            logger.debug("Applying log transformation to endpoint {}", endpoint)
            y_pred = clip_and_log_transform(y_pred)
            y_true = clip_and_log_transform(y_true)

        bootstrap_df = bootstrap_metrics(
            y_pred, y_true, endpoint, ACTIVITY_METRICS, n_bootstrap_samples=BOOTSTRAP_SAMPLES
        )
        all_endpoint_bootstrap_results_list.append(bootstrap_df)
    all_endpoint_bootstrap_results = pd.concat(
        all_endpoint_bootstrap_results_list, ignore_index=True
    )
    all_endpoint_bootstrap_results = all_endpoint_bootstrap_results.fillna(0)
    logger.info("Completed scoring activity predictions")
    return all_endpoint_bootstrap_results


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


def bootstrap_metrics(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    endpoint: str,
    metrics: list[tuple[str, object]],
    n_bootstrap_samples: int,
) -> pd.DataFrame:
    """Calculate bootstrap metrics given predicted and true values.

    Args:
        y_pred (np.ndarray): The predicted values.
        y_true (np.ndarray): The true values.
        endpoint (str): The endpoint for which the metrics are being calculated.
        metrics (list[tuple[str, object]]): List of (metric_name, metric_func) pairs.
        n_bootstrap_samples (int): The number of bootstrap samples to generate.

    Returns:
        pd.DataFrame: A DataFrame containing the bootstrap metrics for the given
                      endpoint.

    """
    bootstrap_metrics_list = []
    for bootstrap_iteration, idx in enumerate(
        bootstrap_sampling(y_true.shape[0], n_bootstrap_samples)
    ):
        metric_values = {"Sample": bootstrap_iteration, "Endpoint": endpoint}
        for metric_name, metric_func in metrics:
            try:
                metric_value = metric_func(y_true[idx], y_pred[idx])
            except Exception as e:
                logger.warning(
                    f"Error calculating metric {metric_name} for endpoint {endpoint}: {e}"
                )
                metric_value = np.nan
            if not isinstance(metric_value, (int, float)):
                metric_value = metric_func(y_true[idx], y_pred[idx]).statistic
            metric_values[metric_name] = metric_value
        bootstrap_metrics_list.append(metric_values)

    bootstrap_df = pd.DataFrame(bootstrap_metrics_list)
    return bootstrap_df


# ---------------------------------------------------------------------------
# TDI (time-dependent inhibition) classification scoring
# ---------------------------------------------------------------------------


def score_tdi_predictions(
    predictions: pd.DataFrame, ground_truth: pd.DataFrame
) -> pd.DataFrame:
    """Score TDI classification predictions against the ground truth.

    ``predictions`` is expected to contain a ``TDI_probability`` column (the
    predicted probability that a compound is TDI-positive); ``ground_truth``
    is expected to contain the binary ``is_TDI`` label. Metrics are computed
    on bootstrapped samples of the dataset, mirroring the activity scoring
    pipeline.

    Args:
        predictions (pd.DataFrame): The predicted TDI probabilities.
        ground_truth (pd.DataFrame): The true TDI labels.

    Returns:
        pd.DataFrame: A DataFrame containing the scored bootstrapped TDI
                      predictions.

    Raises:
        ValueError: If the merged DataFrame contains NaN values after merging
                    predictions with ground truth.

    """
    logger.info("Scoring TDI predictions against ground truth")
    merged_df = predictions.merge(
        ground_truth, on="Molecule Name", suffixes=("_pred", "_true"), how="right"
    ).sort_values("Molecule Name")
    logger.info(
        "Completed merging predictions with ground truth. Merged dataset contains {} "
        "rows and {} columns.",
        merged_df.shape[0],
        merged_df.shape[1],
    )

    if merged_df.isnull().any().any():
        logger.warning(
            "Merged DataFrame contains NaN values after merging predictions with ground"
            " truth. This may indicate missing predictions for some molecules."
        )
        raise ValueError(
            "Merged DataFrame contains NaN values after merging predictions with ground truth."
        )

    y_pred = merged_df["TDI_probability"].to_numpy()
    y_true = merged_df[TDI_ENDPOINT].to_numpy()

    bootstrap_df = bootstrap_metrics(
        y_pred, y_true, TDI_ENDPOINT, TDI_METRICS, n_bootstrap_samples=BOOTSTRAP_SAMPLES
    )
    bootstrap_df = bootstrap_df.fillna(0)
    logger.info("Completed scoring TDI predictions")
    return bootstrap_df
