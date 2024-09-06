import numpy as np


def mean_squared_error(y_true, y_pred):
    """
    Mean squared error regression loss

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) target values

    y_pred : array-like of shape (n_samples,)
        Estimated target values.

    Returns
    -------
    loss : float
        A non-negative floating point value
    """
    loss = np.mean((y_true-y_pred)**2, axis=0)

    return loss