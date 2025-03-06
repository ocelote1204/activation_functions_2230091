import numpy as np

def leaky_relu(x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    """Función de activación Leaky ReLU."""
    return np.maximum(alpha * x, x)

def leaky_relu_derivative(x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    """Derivada de la función Leaky ReLU."""
    return np.where(x > 0, 1, alpha)
