"""Configuration file for the OpenADMET CYP blind challenge."""

import numpy as np
from scipy.stats import kendalltau, spearmanr
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    matthews_corrcoef,
    r2_score,
    roc_auc_score,
)


def rae(y_true, y_pred):
    """Relative absolute error (RAE) metric for regression tasks."""
    return np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true - np.mean(y_true)))


def _binarize(y_pred_proba, threshold: float = 0.5):
    """Threshold predicted TDI probabilities into hard class labels."""
    return (np.asarray(y_pred_proba) >= threshold).astype(int)


def accuracy_from_proba(y_true, y_pred_proba):
    """Accuracy of the TDI classifier after thresholding probabilities at 0.5."""
    return accuracy_score(y_true, _binarize(y_pred_proba))


def balanced_accuracy_from_proba(y_true, y_pred_proba):
    """Balanced accuracy of the TDI classifier after thresholding probabilities at 0.5."""
    return balanced_accuracy_score(y_true, _binarize(y_pred_proba))


def f1_from_proba(y_true, y_pred_proba):
    """F1 score of the TDI classifier after thresholding probabilities at 0.5."""
    return f1_score(y_true, _binarize(y_pred_proba))


def mcc_from_proba(y_true, y_pred_proba):
    """Matthews correlation coefficient after thresholding probabilities at 0.5."""
    return matthews_corrcoef(y_true, _binarize(y_pred_proba))


def roc_auc_safe(y_true, y_pred_proba):
    """ROC-AUC of the predicted TDI probabilities.

    Returns NaN if the bootstrap sample only contains one class, since
    ROC-AUC is undefined in that case.
    """
    if len(np.unique(y_true)) < 2:
        return np.nan
    return roc_auc_score(y_true, y_pred_proba)


# Activity dataset
ENDPOINTS = ["pEC50"]
ENDPOINTS_TO_LOG_TRANSFORM: list[str] = []
ACTIVITY_METRICS = [
    ("MAE", mean_absolute_error),
    ("RAE", rae),
    ("R2", r2_score),
    ("Spearman R", spearmanr),
    ("Kendall's Tau", kendalltau),
]
BOOTSTRAP_SAMPLES = 1000

# TDI (time-dependent inhibition) classification dataset
TDI_ENDPOINT = "is_TDI"
TDI_METRICS = [
    ("Accuracy", accuracy_from_proba),
    ("Balanced Accuracy", balanced_accuracy_from_proba),
    ("F1", f1_from_proba),
    ("MCC", mcc_from_proba),
    ("ROC-AUC", roc_auc_safe),
]
