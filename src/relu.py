import numpy as np

def relu(x: np.ndarray) -> np.ndarray:
    """Función de activación ReLU."""
    return np.maximum(0, x)

def relu_derivative(x: np.ndarray) -> np.ndarray:
    """Derivada de la función ReLU."""
    return np.where(x > 0, 1, 0)
