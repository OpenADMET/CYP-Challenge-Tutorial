"""Utility functions for the OpenADMET CYP blind challenge."""

from functools import lru_cache

import numpy as np

BOOTSTRAP_SEED = 0


def clip_and_log_transform(y: np.ndarray) -> np.ndarray:
    """Clip the input array to zero then apply a log10(y + 1) transformation.

    Args:
        y (np.ndarray): The input array to be transformed.

    Returns:
        np.ndarray: The transformed array.

    """
    y = np.clip(y, a_min=0, a_max=None)
    return np.log10(y + 1)


@lru_cache(maxsize=3)
def bootstrap_sampling(
    original_dataset_size: int, n_bootstrap_repeats: int = 1000
) -> np.ndarray:
    """Generate bootstrap sample indices for a dataset of a given size.

    The random seed is fixed so all submissions are evaluated on the same bootstrap
    samples. Best practices for bootstrap sampling involve sampling the same number of
    samples as the original dataset, with replacement, at least 1000 times.

    Args:
        original_dataset_size (int): The size of the original dataset.
        n_bootstrap_repeats (int): The number of bootstrap samples to generate.
                                   Default is 1000.

    Returns:
        np.ndarray: An array of bootstrap sample indices.

    """
    rng = np.random.default_rng(seed=BOOTSTRAP_SEED)
    return rng.choice(
        original_dataset_size,
        size=(n_bootstrap_repeats, original_dataset_size),
        replace=True,
    )
