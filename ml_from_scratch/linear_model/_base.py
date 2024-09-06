import numpy as np

class LinearRegression:
    def __init__(
        self, 
        fit_intercept = True
    ):
        self.fit_intercept = fit_intercept

    def fit(self, X, y):
        # Prepare dataset
        X_ = np.array(X).copy()
        Y_ = np.array(y).copy()

        # Extract size
        n_samples, n_features = X.shape

        # Create the design matrix 
        if self.fit_intercept:
            A = np.column_stack((X, np.ones(n_samples)))
        else:
            A = X

        # Solve for model parameter, beta
        beta = np.linalg.inv(A.T @ A) @ A.T @ y

        # Extract model parameter
        if self.fit_intercept:
            self.coef_ = beta[:n_features]
            self.intercept_ = beta[-1]
        else:
            self.coef_ = beta
            self.intercept_ = 0

    def predict(self, X):
        X = np.array(X)
        y_pred = np.dot(X, self.coef_) + self.intercept_

        return y_pred