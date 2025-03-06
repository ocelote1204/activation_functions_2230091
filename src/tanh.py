import numpy as np

def tanh(x: np.ndarray) -> np.ndarray:
    """Función de activación tangente hiperbólica."""
    return np.tanh(x)

def tanh_derivative(x: np.ndarray) -> np.ndarray:
    """Derivada de la función tangente hiperbólica."""
    return 1 - np.tanh(x)**2
