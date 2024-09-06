import numpy as np


# CLASSIFICATION IMPURITY
def Gini(y):
    """
    Calculate gini index
    from ESL (9.17)

    Parameter
    ---------
    y : {array-like} of shape (N_m,)
        The output of data in region R_m

    Return
    ------
    node_impurity : float
        The node impurity
    """
    # Extract class
    K, counts = np.unique(y, return_counts=True)
    unique_counts = dict(zip(K, counts))
    N_m = len(y)

    # Calculate the proportion of class k observations in node m
    p_m = {}
    for k in K:
        p_m[k] = unique_counts[k] / N_m

    # Calculate the node impurity
    node_impurity = 0
    for k in K:
        node_impurity += p_m[k] * (1-p_m[k])

    return node_impurity

def Log_Loss(y):
    """
    Calculate log loss (missclassification error)
    from ESL (9.17)

    Parameter
    ---------
    y : {array-like} of shape (N_m,)
        The output of data in region R_m

    Return
    ------
    node_impurity : float
        The node impurity
    """
    # Extract class
    K, counts = np.unique(y, return_counts=True)
    unique_counts = dict(zip(K, counts))
    N_m = len(y)

    # Calculate the proportion of class k observations in node m
    p_m = {}
    for k in K:
        p_m[k] = unique_counts[k] / N_m

    # Find the majority class in node m
    ind_max = np.argmax(counts)
    class_max = K[ind_max]

    # Calculate the node impurity
    node_impurity = 1 - p_m[class_max]

    return node_impurity

def Entropy(y):
    """
    Calculate Cross-entropy or deviance
    from ESL (9.17)

    Parameter
    ---------
    y : {array-like} of shape (N_m,)
        The output of data in region R_m

    Return
    ------
    node_impurity : float
        The node impurity
    """
    # Extract class
    K, counts = np.unique(y, return_counts=True)
    unique_counts = dict(zip(K, counts))
    N_m = len(y)

    # Calculate the proportion of class k observations in node m
    p_m = {}
    for k in K:
        p_m[k] = unique_counts[k] / N_m

    # Calculate the node impurity
    node_impurity = 0
    for k in K:
        node_impurity += p_m[k] * np.log(p_m[k])

    return node_impurity


# REGRESSION IMPURITY
def MSE(y):
    """
    Calculate Mean squared error
    from ESL (9.15)

    Parameter
    ---------
    y : {array-like} of shape (N_m,)
        The output of data in region R_m

    Return
    ------
    node_impurity : float
        The node impurity
    """
    # Extract number of samples
    N_m = len(y)

    # Calculate the best prediction (c) --> Eq. (9.11)
    c_m = (1/N_m) * np.sum(y)

    # Calculate the node-impurity (variance)
    node_impurity = 0
    for i in range(N_m):
        node_impurity += (y[i] - c_m)**2

    node_impurity *= (1/N_m)

    return node_impurity

def MAE(y):
    """
    Calculate Mean absolute error
    from ESL (9.15)

    Parameter
    ---------
    y : {array-like} of shape (N_m,)
        The output of data in region R_m

    Return
    ------
    node_impurity : float
        The node impurity
    """
    # Extract number of samples
    N_m = len(y)

    # Calculate the best prediction (c) --> Eq. (9.11)
    c_m = np.median(y)

    # Calculate the node-impurity (variance)
    node_impurity = 0
    for i in range(N_m):
        node_impurity += np.abs(y[i] - c_m)

    node_impurity *= (1/N_m)

    return node_impurity
