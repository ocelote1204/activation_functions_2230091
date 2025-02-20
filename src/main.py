import numpy as np
import matplotlib.pyplot as plt

def relu(x: np.ndarray) -> np.ndarray:
    """
    Función de activación ReLU (Rectified Linear Unit).
    
    Aplica la función ReLU a cada elemento de la entrada. La salida es 0 para valores negativos y 
    el valor de entrada para valores positivos.
    
    Args:
        x: Un array de valores numéricos.
        
    Returns:
        Un array con los valores transformados por la función ReLU.
    """
    return np.maximum(0, x)

def relu_derivative(x: np.ndarray) -> np.ndarray:
    """
    Derivada de la función de activación ReLU.
    
    La derivada de ReLU es 0 cuando el valor de entrada es menor o igual a 0 y 1 cuando es mayor que 0.
    
    Args:
        x: Un array de valores numéricos.
        
    Returns:
        Un array con los valores de la derivada de ReLU.
    """
    return np.where(x > 0, 1, 0)

def sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Función de activación sigmoide.
    
    Aplica la función sigmoide a cada elemento de la entrada. La salida es un valor entre 0 y 1.
    
    Args:
        x: Un array de valores numéricos.
        
    Returns:
        Un array con los valores transformados por la función sigmoide.
    """
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x: np.ndarray) -> np.ndarray:
    """
    Derivada de la función sigmoide.
    
    La derivada de la función sigmoide se calcula como sigmoide(x) * (1 - sigmoide(x)).
    
    Args:
        x: Un array de valores numéricos.
        
    Returns:
        Un array con los valores de la derivada de la sigmoide.
    """
    sig = sigmoid(x)
    return sig * (1 - sig)

def tanh(x: np.ndarray) -> np.ndarray:
    """
    Función de activación tangente hiperbólica (tanh).
    
    Aplica la función tangente hiperbólica a cada elemento de la entrada. La salida es un valor entre -1 y 1.
    
    Args:
        x: Un array de valores numéricos.
        
    Returns:
        Un array con los valores transformados por la función tanh.
    """
    return np.tanh(x)

def tanh_derivative(x: np.ndarray) -> np.ndarray:
    """
    Derivada de la función tangente hiperbólica (tanh).
    
    La derivada de tanh es 1 - tanh(x)^2.
    
    Args:
        x: Un array de valores numéricos.
        
    Returns:
        Un array con los valores de la derivada de la tangente hiperbólica.
    """
    return 1 - np.tanh(x)**2

def leaky_relu(x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    """
    Función de activación Leaky ReLU.
    
    La función Leaky ReLU aplica una pequeña pendiente (alpha) para los valores negativos y la 
    función ReLU estándar para los valores positivos.
    
    Args:
        x: Un array de valores numéricos.
        alpha: Factor de pendiente para los valores negativos (por defecto es 0.01).
        
    Returns:
        Un array con los valores transformados por la función Leaky ReLU.
    """
    return np.maximum(alpha * x, x)

def leaky_relu_derivative(x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    """
    Derivada de la función Leaky ReLU.
    
    La derivada de Leaky ReLU es 1 para los valores positivos y alpha para los negativos.
    
    Args:
        x: Un array de valores numéricos.
        alpha: Factor de pendiente para los valores negativos (por defecto es 0.01).
        
    Returns:
        Un array con los valores de la derivada de la función Leaky ReLU.
    """
    return np.where(x > 0, 1, alpha)

def softmax(x: np.ndarray) -> np.ndarray:
    """
    Función de activación Softmax.
    
    Softmax convierte un vector de valores en probabilidades, sumando 1 a través de todas las salidas.
    
    Args:
        x: Un array unidimensional de valores numéricos.
        
    Returns:
        Un array con los valores transformados por la función Softmax.
    """
    e_x = np.exp(x - np.max(x))  # Subtract max(x) for numerical stability
    return e_x / e_x.sum(axis=0)

def swish(x: np.ndarray) -> np.ndarray:
    """
    Función de activación Swish.
    
    La función Swish es una combinación de la entrada multiplicada por la función sigmoide.
    
    Args:
        x: Un array de valores numéricos.
        
    Returns:
        Un array con los valores transformados por la función Swish.
    """
    return x * sigmoid(x)

# Crear un rango de valores para x
x = np.linspace(-10, 10, 400)

# **ReLU y su derivada**
plt.figure(figsize=(10, 8))
plt.subplot(2, 1, 1)
plt.plot(x, relu(x), label='ReLU', color='#1f77b4', linewidth=2)
plt.title('ReLU', fontsize=16)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(x, relu_derivative(x), label='Derivada de ReLU', color='#d62728', linewidth=2)
plt.title('Derivada de ReLU', fontsize=16)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

plt.tight_layout()
plt.show()

# **Sigmoide y su derivada**
plt.figure(figsize=(10, 8))
plt.subplot(2, 1, 1)
plt.plot(x, sigmoid(x), label='Sigmoide', color='#2ca02c', linewidth=2)
plt.title('Sigmoide', fontsize=16)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(x, sigmoid_derivative(x), label='Derivada de Sigmoide', color='#9467bd', linewidth=2)
plt.title('Derivada de Sigmoide', fontsize=16)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

plt.tight_layout()
plt.show()

# **Tanh y su derivada**
plt.figure(figsize=(10, 8))
plt.subplot(2, 1, 1)
plt.plot(x, tanh(x), label='Tanh', color='#ff7f0e', linewidth=2)
plt.title('Tanh', fontsize=16)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(x, tanh_derivative(x), label='Derivada de Tanh', color='#8c564b', linewidth=2)
plt.title('Derivada de Tanh', fontsize=16)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

plt.tight_layout()
plt.show()

# **Leaky ReLU y su derivada**
plt.figure(figsize=(10, 8))
plt.subplot(2, 1, 1)
plt.plot(x, leaky_relu(x), label='Leaky ReLU', color='#17becf', linewidth=2)
plt.title('Leaky ReLU', fontsize=16)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(x, leaky_relu_derivative(x), label='Derivada de Leaky ReLU', color='#e377c2', linewidth=2)
plt.title('Derivada de Leaky ReLU', fontsize=16)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

plt.tight_layout()
plt.show()

# **Swish**
plt.figure(figsize=(10, 8))
plt.plot(x, swish(x), label='Swish', color='#f29e29', linewidth=2)
plt.title('Swish', fontsize=16)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

plt.tight_layout()
plt.show()

# **Softmax (Visualizado sobre un vector)**
x_softmax = np.linspace(-5, 5, 10)
plt.figure(figsize=(10, 8))
plt.plot(x_softmax, softmax(x_softmax), label='Softmax', color='#9467bd', linewidth=2)
plt.title('Softmax', fontsize=16)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

plt.tight_layout()
plt.show()
