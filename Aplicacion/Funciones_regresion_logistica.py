import numpy as np

def sigmoid(z):
    """
    Función sigmoide para la regresión logística.
    """
    return 1 / (1 + np.exp(-z))

def estimator_fn(y, X, bias=True, max_iter=100, tol=0.0001, lambda_=1.0):
    """
    Estima los pesos de la regresión logística con regularización L2.
    
    Args:
    y: array, target.
    X: array, features.
    bias: bool, si True, se incluye el término de sesgo (intercept).
    max_iter: int, número máximo de iteraciones.
    tol: float, tolerancia para la convergencia.
    lambda_: float, parámetro de regularización.
    
    Returns:
    weights: array, pesos estimados.
    """
    if bias:
        X_ = np.c_[np.ones(shape=(len(y), 1)), X]
    else:
        X_ = X
    
    # Inicializar los pesos
    weights = np.zeros(X_.shape[1])
    
    # Iterar hasta que se alcance la convergencia
    for _ in range(max_iter):
        z = np.dot(X_, weights)
        p = sigmoid(z)
        W = np.diag(p * (1 - p))
        H = np.dot(X_.T, np.dot(W, X_)) + lambda_ * np.eye(X_.shape[1])
        gradient = np.dot(X_.T, y - p) - lambda_ * weights
        delta = np.linalg.solve(H, gradient)
        weights += delta
        
        # Verificar la convergencia
        if np.linalg.norm(delta) < tol:
            break
    
    return weights



def predictor_fn(X, weights, bias=True):
    """
    Predice las probabilidades utilizando el modelo de regresión logística.
    
    Args:
    X: array, features.
    weights: array, pesos del modelo de regresión logística.
    bias: bool, si True, se incluye el término de sesgo (intercept).
    
    Returns:
    probs: array, probabilidades predichas.
    """
    if bias:
        X_ = np.c_[np.ones(shape=(len(X), 1)), X]
    else:
        X_ = X
    z = np.dot(X_, weights)
    probs = sigmoid(z)
    return probs
