import numpy as np
from activaciones.sigmoid import sigmoid

def swish(x: np.ndarray) -> np.ndarray:
    """Función de activación Swish."""
    return x * sigmoid(x)
