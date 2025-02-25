import numpy as np

def softmax(x: np.ndarray) -> np.ndarray:
    """Función de activación Softmax."""
    e_x = np.exp(x - np.max(x))  # Subtract max(x) for numerical stability
    return e_x / e_x.sum(axis=0)
