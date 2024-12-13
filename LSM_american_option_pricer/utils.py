import numpy as np
from numpy.random import default_rng
from scipy.special import eval_laguerre
from sklearn import linear_model
from sklearn.linear_model import Ridge
import time


def scale_X(X):
    '''
    X: np.ndarray simulated asset price paths, shape(N, M+1)
    compute the min-max scaling data across all path
    '''
    min_ = np.min(X)
    max_ = np.max(X)
    if max_ - min_ == 0:
        return np.zeros_like(X)
    return (X - min_) / (max_ - min_)


def BasisFunctLaguerre(X, k=3):
    """
    Generate Laguerre basis functions for input data X.
    Parameters:
    - X (numpy.ndarray): Input data of shape (n,) or (n, d).
    - k (int): Number of Laguerre basis functions to generate (degrees 1 to k).
    Returns:
    - numpy.ndarray: A matrix of shape (n, k) where each column corresponds
                     to a Laguerre polynomial of degree 1 to k applied to X.
    """
    X = np.asarray(X)

    # Ensure X is a 2D array (n, d). If 1D, reshape to (n, 1)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    # min-max scaling to ensure computation stability

    X = scale_X(X)

    n, d = X.shape
    basis = np.zeros((n, k))

    for degree in range(1, k + 1):
        if d == 1:
            basis[:, degree - 1] = eval_laguerre(degree, X[:, 0])
        else:
            # If multiple dimensions, concatenate basis functions for each feature
            # This will increase the number of columns accordingly
            for feature in range(d):
                basis[:, (degree - 1) * d + feature] = eval_laguerre(degree, X[:, feature])
    return basis


def computeBetaReg(feature_arr, discounted_arr, alpha=1.0):
    """
    Compute Ridge regression coefficients.

    Parameters:
    - feature_arr (numpy.ndarray): Feature matrix of shape (n_samples, n_features).
    - discounted_arr (numpy.ndarray): Target vector of shape (n_samples,).
    - alpha (float): Regularization strength. Must be a positive float. Regularization improves the conditioning of the problem and reduces the variance of the estimates.

    Returns:
    - numpy.ndarray: Regression coefficients of shape (n_features,).
    """
    ridge_reg = Ridge(alpha=alpha, fit_intercept=True)
    ridge_reg.fit(feature_arr, discounted_arr)
    coefficients = ridge_reg.coef_
    return coefficients