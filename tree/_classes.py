"""
This module gathers tree-based methods
"""

import numpy as np

from . import _criterion


# =============================================================================
# Types and constants
# =============================================================================
CRITERIA_CLF = {
    "gini": _criterion.Gini,
    "log_loss": _criterion.Log_Loss,
    "entropy": _criterion.Entropy
}

CRITERIA_REG = {
    "squared_error": _criterion.MSE,
    "absolute_error": _criterion.MAE
}

# =============================================================================
# Base decision tree
# =============================================================================

def _split_data(data, feature, threshold):
    """
    Split data based on given feature and threshold
    
    Parameters
    ----------
    data : {array-like}, shape of (n_samples, n_features+1)
        sample data X, y

    feature: str
        feature to split

    threshold: float
        threshold to split the data
        if data[feature] > threshold
            return data_right
        else:
            return data_left

    Returns
    -------
    data_left: {array-like}, shape of (n_samples_1, n_features+1)
        X, y data that its X[feature] <= threshold

    data_right: {array-like}, shape of (n_samples_2, n_features+1)
        X, y data that its X[feature] > threshold
    """
    cond_left = data[:, feature] <= threshold
    data_left = data[cond_left]
    data_right = data[~cond_left]

    return data_left, data_right

def _generate_possible_split(data):
    """
    Generate possible split threshold
    """
    # Copy data
    data = data.copy()

    # Extract the unique value
    unique_val = np.unique(data)

    # Extract shape of unique_val
    m = len(unique_val)

    # Sort data
    unique_val.sort()

    # Initialize threshold
    threshold = np.zeros(m-1)

    # Create the possible split
    for i in range(m-1):
        val_1 = unique_val[i]
        val_2 = unique_val[i+1]

        threshold[i] = 0.5*(val_1 + val_2)

    return threshold

def _calculate_majority_vote(y):
    """
    Calculate the majority vote in the output data
    Use this if the task is classification

    Parameter
    ---------
    y : {array-like} of shape (n_samples,)
        The output samples data

    Return
    ------
    y_pred : int or str
        The most common class
    """
    # Extract output
    vals, counts = np.unique(y, return_counts = True)

    # Find the majority vote
    ind_max = np.argmax(counts)
    y_pred = vals[ind_max]

    return y_pred

def _calculate_average_vote(y):
    """
    Calculate the average vote in the output data
    Use this if the task is regression

    Parameter
    ---------
    y : {array-like} of shape (n_samples,)
        The output samples data

    Return
    ------
    y_pred : int or str
        The average of the output
    """
    y_pred = np.mean(y)
    return y_pred

def _to_string(tree, indent="|   "):
    """
    A function to print the decision tree recursively

    Parameters
    ----------
    tree : object
        The tree object

    indent : str, default="|   "
        The indentation

    Returns
    -------
    text_to_print : str
        The text to print
    """
    if tree.is_leaf:
        # If it is leaf, print the predicted value
        text_to_print = f'Pred: {tree.value:.2f}'
        
    else:
        # If it is not a leaf, print the branch
        decision = f"feature_{tree.feature} <= {tree.threshold:.2f}?"

        # Print the true branch recursively
        true_branch = indent + '|T: ' + _to_string(tree = tree.children_left,
                                                   indent = indent + '|   ')
        
        # Print the false branch recursively
        false_branch = indent + '|F: ' + _to_string(tree = tree.children_right,
                                                    indent = indent + '|   ')
        
        # Summarize
        text_to_print = decision + '\n' + true_branch + '\n' + false_branch
    
    return text_to_print


class Tree:
    """
    Object-based representation of a binary decision tree.

    Parameters
    ----------
    feature : str, default=None
        Node feeature to split on

    threshold : float, default=None
        Threshold for the internal node i

    value : float, default=None
        Containts the constant prediction value of each node

    impurity : float, default=None
        Holds the impurity (i.e., the value of the splitting criterion)
        at node i

    children_left : Tree object, default=None
        Handles the case where X[:, feature[i]] <= threshold[i]

    children_right : Tree object, default=None
        Handles the case where X[:, feature[i]] > threshold[i]

    is_leaf : bool, deafult=False
        Whether the current node is a leaf or not

    n_samples : int, default=None
        The number of samples in current node
    """
    def __init__(
        self,
        feature=None,
        threshold=None,
        value=None,
        impurity=None,
        children_left=None,
        children_right=None,
        is_leaf=False,
        n_samples=None
    ):
        self.feature = feature
        self.threshold = threshold
        self.value = value
        self.impurity = impurity
        self.children_left = children_left
        self.children_right = children_right
        self.is_leaf = is_leaf
        self.n_samples = n_samples

class BaseDecisionTree:
    """
    Base class for decision trees

    Warning: This class should not be used directly.
    Use derived classes instead
    """
    def __init__(
        self,
        criterion,
        max_depth,
        min_samples_split,
        min_samples_leaf,
        min_impurity_decrease,
        alpha=0.0
    ):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.alpha = alpha

    def _best_split(self, X, y):
        """
        Find the best split for a node
        """
        # Need at least min_samples_split to split a node
        m = len(y)
        if m < self.min_samples_split:
            return None, None
        
        # Initialize
        parent = np.column_stack((X, y))
        best_gain = 0.0
        best_feature, best_threshold = None, None
        for feature_i in range(self.n_features):
            # Extract data of selected feature
            X_i = X[:, feature_i]

            # Find the possible split threshold
            threshold = _generate_possible_split(data = X_i)

            # Iterate over threshold to find the best split
            for i in range(len(threshold)):
                # Split the parent
                left_children, right_children = _split_data(data = parent,
                                                            feature = feature_i,
                                                            threshold = threshold[i])
                
                # Extract the children's output
                left_y = left_children[:, self.n_features:]
                right_y = right_children[:, self.n_features:]

                # Calculate the impurity decrease (gain)
                cond_1 = len(left_y) >= self.min_samples_leaf
                cond_2 = len(right_y) >= self.min_samples_leaf
                if cond_1 and cond_2:
                    current_gain = self._calculate_impurity_decrease(y,
                                                                     left_y,
                                                                     right_y)

                    if current_gain > best_gain:
                        best_gain = current_gain
                        best_feature = feature_i
                        best_threshold = threshold[i]

        if best_gain >= self.min_impurity_decrease:
            return best_feature, best_threshold
        else:
            return None, None

    def _calculate_impurity_decrease(self, parent, left_children, right_children):
        """
        Calculate the impurity decrease
        The weighted impurity decrease equation is the following::
        
            N_t/N * (
                parent_impurity
                - (N_t_R / N_T) * right_child_impurity
                - (N_t_L / N_T) * left_child_impurity
            )

        where
        - N     : total number of samples
        - N_t   : the number of samples at the current node
        - N_t_L : the number of samples in the left child
        - N_t_R : the number of samples in the right child

        Parameters
        ----------
        parent : {array-like} of shape (N_t,)
            output parent node

        left_children: {array-like} of shape (N_t_L,)
            output child left node

        right_children: {array-like} of shape (N_t_R,)
            output child right node

        impurity: function
            the impurity solver based on criterion

        Return
        ------
        impurity_decrease : float
            The wieghted impurity decrease
        """
        # Calculate the number of samples
        N = self.n_samples
        N_T = len(parent)
        N_t_L = len(left_children)
        N_t_R = len(right_children)

        # Calculate the impurity
        I_parent = self._impurity_evaluation(parent)
        I_child_left = self._impurity_evaluation(left_children)
        I_child_right = self._impurity_evaluation(right_children)

        # Calculate the weighted impurity
        impurity_decrease = I_parent \
                            - (N_t_R / N_T) * I_child_right \
                            - (N_t_L / N_T) * I_child_left
        
        impurity_decrease *= (N_T/N)

        return impurity_decrease        

    def _grow_tree(self, X, y, depth=0):
        """
        Build a decision tree by recursively finding the best split
        """
        # Create node (whether it's internal or leaves)
        node_impurity = self._impurity_evaluation(y)
        node_value = self._leaf_value_calculation(y)
        node = Tree(
            value = node_value,
            impurity = node_impurity,
            is_leaf = True,
            n_samples = len(y)
        )

        # Split recursively until maximum depth is reached
        if self.max_depth is None:
            cond = True
        else:
            cond = depth < self.max_depth

        if cond:
            # Find the best split
            feature_i, threshold_i = self._best_split(X, y)

            if feature_i is not None:
                # Split the data
                data = np.column_stack((X, y))
                data_left, data_right = _split_data(data = data,
                                                    feature = feature_i,
                                                    threshold = threshold_i)
                
                # Extract X and y
                X_left = data_left[:, :self.n_features]
                y_left = data_left[:, self.n_features:]
                X_right = data_right[:, :self.n_features]
                y_right = data_right[:, self.n_features:]

                # Grow the tree
                node.feature = feature_i
                node.threshold = threshold_i
                node.children_left = self._grow_tree(X_left, y_left, depth+1)
                node.children_right = self._grow_tree(X_right, y_right, depth+1)
                node.is_leaf = False

        return node

    def _prune_tree(self, tree=None):
        """
        This is a function to prune a tree
        
        Notes
        -----
        The CART algorithm uses minimum complexity cost to prune a tree.
        However, it is hard to implement right now, so I changed it to impurity gain.
            
            if current_gain(tree, sub_tree) < self.alpha:
                prune the sub_tree
                make the tree as a leaf

        """
        if not tree:
            tree = self.tree_

        if tree.is_leaf:
            pass
        else:
            self._prune_tree(tree.children_left)
            self._prune_tree(tree.children_right)

            # merge potentially
            if tree.children_right.is_leaf == False and tree.children_left.is_leaf == False:
                n_true = tree.children_left.n_samples
                n_false = tree.children_right.n_samples

                p = n_true / (n_true + n_false)
                delta = tree.impurity - p*tree.children_left.impurity - (1-p)*tree.children_right.impurity
                if delta < self.alpha:
                    tree.children_left, tree.children_right = None, None
                    tree.threshold = None
                    tree.feature = None
                    tree.is_leaf = True

    def _export_tree(self):
        """
        Function to call _to_string method
        """
        print("The Decision Tree")
        print("-----------------")
        print(_to_string(tree=self.tree_))

    def _predict_value(self, X, tree=None):
        """
        Predict the value by following the tree recursively

        Parameters
        ----------
        X : {array-like} of shape (1, n_features)
            The sample input

        tree : Tree object, default=None
            The tree object
        """
        # Define tree if there is any
        if tree is None:
            tree = self.tree_

        # Check whether it is a leaf or not
        if tree.is_leaf:
            # Return the predicted value if it is a leaf
            return tree.value
        else:        
            # Beside that, it is a branch, so you choose the feature to track
            feature_value = X[:, tree.feature]

            # Then, determine which branch to follow
            if feature_value <= tree.threshold:
                branch = tree.children_left
            else:
                branch = tree.children_right

            return self._predict_value(X, branch)

    def fit(self, X, y):
        """
        Build a decision tree using CART algorithm
        1. Grow the tree
        2. Prune the tree
        """
        # Convert the input
        X = np.array(X).copy()
        y = np.array(y).copy()

        # Extract the data shape
        self.n_samples, self.n_features = X.shape

        # Grow tree
        self.tree_ = self._grow_tree(X, y)

        # Prune tree
        self._prune_tree()

    def predict(self, X):
        """
        Predict the value with Decision Tree

        Parameters
        ----------
        X : {array-like} of shape (n_samples, n_features)
            The sample input

        Return
        ------
        y : {array-like} of shape (n_samples,)
            The predicted value
        """
        # Convert input data
        X = np.array(X).copy()

        # Predict
        y = [self._predict_value(sample.reshape(1, -1)) for sample in X]

        return y

class DecisionTreeClassifier(BaseDecisionTree):
    """
    A decision tree classifier
    Use CART algorithm

    Parameters
    ----------
    criterion : {"gini", "entropy", "log_loss"}, default="gini"
        The function to measure the quality of a split.

    max_depth : int, default=None
        The maximum depth of the tree. If None, the nodes are expanded
        until all leaves are pure or until all leaves contain less than
        min_samples_split samples

    min_samples_split : int, default=2
        The minimum number of samples required to split an internal node

    min_impurity_decrease : float, default=0.0
        A node will be split if this split induces a decrease of the impurity
        greater than or equal to this value.

        ref: sklearn.tree.DecisionTreeClassifier
        The weighted impurity decrease equation is the following::
        
            N_t/N * (
                parent_impurity
                - (N_t_R / N_T) * right_child_impurity
                - (N_t_L / N_T) * left_child_impurity
            )

        where
        - N     : total number of samples
        - N_t   : the number of samples at the current node
        - N_t_L : the number of samples in the left child
        - N_t_R : the number of samples in the right child
    """
    def __init__(
        self,
        criterion="gini",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_impurity_decrease=0.0,
        alpha=0.0
    ):
        super().__init__(
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_impurity_decrease=min_impurity_decrease,
            alpha=alpha
        )

    def fit(self, X, y):
        """
        Build a decision tree classifier

        Parameters
        ----------
        X : {array-like} of shape (n_samples, n_features)
            The training input samples

        y : {array-like} of shape (n_samples,)

        Returns
        -------
        self : DecisionTreeClassifier
            Fitted estimator
        """
        # Initialize solver
        self._impurity_evaluation = CRITERIA_CLF[self.criterion]
        self._leaf_value_calculation = _calculate_majority_vote

        super(DecisionTreeClassifier, self).fit(X, y)

class DecisionTreeRegressor(BaseDecisionTree):
    """
    A decision tree regressor
    Use CART algorithm

    Parameters
    ----------
    criterion : {"squared_error", "absolute_error"}, default="squared_error"
        The function to measure the quality of a split.

    max_depth : int, default=None
        The maximum depth of the tree. If None, the nodes are expanded
        until all leaves are pure or until all leaves contain less than
        min_samples_split samples

    min_samples_split : int, default=2
        The minimum number of samples required to split an internal node

    min_impurity_decrease : float, default=0.0
        A node will be split if this split induces a decrease of the impurity
        greater than or equal to this value.

        ref: sklearn.tree.DecisionTreeRegressor
        The weighted impurity decrease equation is the following::
        
            N_t/N * (
                parent_impurity
                - (N_t_R / N_T) * right_child_impurity
                - (N_t_L / N_T) * left_child_impurity
            )

        where
        - N     : total number of samples
        - N_t   : the number of samples at the current node
        - N_t_L : the number of samples in the left child
        - N_t_R : the number of samples in the right child
    """
    def __init__(
        self,
        criterion="squared_error",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_impurity_decrease=0.0,
        alpha=0.0
    ):
        super().__init__(
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_impurity_decrease=min_impurity_decrease,
            alpha=alpha
        )

    def fit(self, X, y):
        """
        Build a decision tree regressor

        Parameters
        ----------
        X : {array-like} of shape (n_samples, n_features)
            The training input samples

        y : {array-like} of shape (n_samples,)

        Returns
        -------
        self : DecisionTreeRegressor
            Fitted estimator
        """
        # Initialize solver
        self._impurity_evaluation = CRITERIA_REG[self.criterion]
        self._leaf_value_calculation = _calculate_average_vote

        super(DecisionTreeRegressor, self).fit(X, y)
