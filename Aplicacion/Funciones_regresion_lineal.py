import numpy as np

def estimator_fn(y, X, bias=True):
    if bias:
        X_ = np.c_[np.ones(shape=(len(y), 1)), X]
    else:
        X_ = X
    # OLS by LU decomposition
    return np.linalg.solve(np.dot(X_.T, X_), np.dot(X_.T, y))



def predictor_fn(X, weights, bias=True):
    if bias:
        X_ = np.c_[np.ones(shape=(len(X), 1)), X]
    else:
        X_ = X
    return np.dot(X_, weights)


def loss_fn(y, y_pred):
    return np.mean((y - y_pred)**2)