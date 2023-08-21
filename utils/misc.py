import numpy as np


def min_max_normalization(arr: np.ndarray):
    """
    Min-Max Normalization.
    arr - arr.min() / (arr.max() - arr.min() + 1e-8)

    Args:
        arr (np.ndarray): array to normalize
    Returns:
        arr (np.ndarray): normalized array
    """
    return (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)


def standardization(arr: np.ndarray):
    """
    Standardization.
    arr - arr.mean() / arr.std()

    Args:
        arr (np.ndarray): array to standardize
    Returns:
        arr (np.ndarray): standardized array
    """
    return (arr - arr.mean()) / (arr.std() + 1e-8)
