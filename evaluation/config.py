"""Configuration file for the OpenADMET CYP blind challenge.

Ported from the challenge backend, so it should match exactly what you see on the leaderboard!
"""

from functools import partial

from scipy.stats import kendalltau, spearmanr
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
)

from .custom_scoring_functions import rae_soft_threshold_absolute_error

# Multi-endpoint macro-averaging .
# A pseudo-endpoint, scored alongside the real endpoints in every bootstrap sample, whose
# per-metric values are macro-averages across endpoints rather than raw per-endpoint scores.
MACRO_ENDPOINT_LABEL = "MA"

# Activity dataset
IDENTIFIER_COLUMNS = ["SMILES", "Molecule_Name"]
REGRESSION_ENDPOINTS = [
    "CYP1A2_pIC50_direct_inhibition",
    "CYP2C9_pIC50_direct_inhibition",
    "CYP2D6_pIC50_direct_inhibition",
    "CYP3A4_pIC50_direct_inhibition",
]
REGRESSION_CREDIBLE_INTERVALS_UPPER_SUFFIX = "_conf_high"
REGRESSION_CREDIBLE_INTERVALS_LOWER_SUFFIX = "_conf_low"
# Leave list empty if no classification tasks
CLASSIFICATION_ENDPOINTS = [
    "CYP2D6_is_TDI",
    "CYP3A4_is_TDI",
]
ACTIVITY_ENDPOINTS = REGRESSION_ENDPOINTS + CLASSIFICATION_ENDPOINTS
ENDPOINTS_TO_LOG_TRANSFORM: list[str] = []
ACTIVITY_DATASET_SIZE = 750
ACTIVITY_METRICS = [
    ("ST-RAE", rae_soft_threshold_absolute_error),
    ("MAE", mean_absolute_error),
    ("R2", r2_score),
    ("Spearman_R", spearmanr),
    ("Kendall_Tau", kendalltau),
]
# Rank correlations (Spearman_R, Kendall_Tau) are mathematically undefined  
# whenever a bootstrap sample has zero variance in
# y_pred (e.g. a submission that predicts the same value for every compound) or y_true. 
# 0.0 is the "no correlation" value on both metrics' [-1, 1] scale, matching how an
# unconditionally-constant predictor should be scored: no better than chance, not a
# hard failure. 
METRIC_NAN_FALLBACK: dict[str, float] = {
    "Spearman_R": 0.0,
    "Kendall_Tau": 0.0,
}
# zero_division=0 matches sklearn's documented degenerate-case default, avoiding
# warnings/errors on bootstrap resamples with no positive predictions (TDI labels are
# imbalanced). matthews_corrcoef already returns 0.0 (not NaN) in its own degenerate
# case, so it needs no wrapping.
CLASSIFICATION_METRICS = [
    ("MCC", matthews_corrcoef),
    ("Accuracy", accuracy_score),
    ("Precision", partial(precision_score, zero_division=0)),
    ("Recall", partial(recall_score, zero_division=0)),
    ("F1", partial(f1_score, zero_division=0)),
]
SORT_REGRESSION_LEADERBOARD_BY = "ST-RAE"
SORT_CLASSIFICATION_LEADERBOARD_BY = "MCC"
BOOTSTRAP_SAMPLES = 1000
