import numpy as np

def sigmoid(x: np.ndarray) -> np.ndarray:
    """Función de activación sigmoide."""
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x: np.ndarray) -> np.ndarray:
    """Derivada de la función sigmoide."""
    sig = sigmoid(x)
    return sig * (1 - sig)
