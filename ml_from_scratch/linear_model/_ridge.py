import numpy as np

from ._base import LinearRegression

class Ridge(LinearRegression):
    def __init__(
        self,
        alpha=1.0,
        fit_intercept=True
    ):
        self.aplha = alpha
        self.fit_intercept = fit_intercept

    def fit(self, X, y):
        X = np.array(X).copy()
        Y = np.array(y).copy()
        
        n_samples, n_features = X.shape
       
        if self.fit_intercept:
            A = np.column_stack((X, np.ones(n_samples)))
        else:
            A = X.copy()

        XT_X = A.T @ A
        XT_Y = A.T @ Y
        if self.fit_intercept:
            alpha_I = alpha*np.identity(A.shape[1])
            alpha_I[-1, -1] = 0.0
        else:
            alpha_I = alpha*np.identity(A.shape[1])

        beta = np.linalg.inv(XT_X + alpha_I) @ XT_Y

        if self.fit_intercept:
            self.coef_ = beta[:-1]
            self.intercept_ = beta[-1]
        else:
            self.coef_ = beta.copy()
            self.intercept = 0.0