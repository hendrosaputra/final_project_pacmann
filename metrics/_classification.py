import numpy as np


def accuracy_score(y_true, y_pred):
    """Accuracy classification score.
    
    Parameters
    ----------
    y_true : 1d array-like, or label indicator array / sparse matrix
        Ground truth (correct) labels.
    
    y_pred : 1d array-like, or label indicator array / sparse matrix
        Predicted labels, as returned by a classifier.
    
    Returns
    -------
    score : float
        The accuracy score
        # correct prediction / # total data
    
    Examples
    --------
    >>> from sklearn.metrics import accuracy_score
    >>> y_pred = [0, 2, 1, 3]
    >>> y_true = [0, 1, 2, 3]
    >>> accuracy_score(y_true, y_pred)
    0.5
    """
    # Compute accuracy score
    n_true = np.sum(y_true == y_pred)
    n_total = len(y_true)

    return n_true/n_total
