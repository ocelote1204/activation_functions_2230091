import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Agregar 'src/' al path para poder importar los módulos
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from activaciones import (
    relu, relu_derivative, sigmoid, sigmoid_derivative, 
    tanh, tanh_derivative, leaky_relu, leaky_relu_derivative, 
    softmax, swish
)

def plot_function_and_derivative(x, func, deriv, func_name, color_func, color_deriv):
    """Grafica una función de activación y su derivada en subgráficas dentro de una misma ventana."""
    plt.subplot(2, 1, 1)  # Subgráfico 1
    plt.plot(x, func(x), label=func_name, color=color_func, linewidth=2)
    plt.title(func_name, fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    plt.subplot(2, 1, 2)  # Subgráfico 2
    plt.plot(x, deriv(x), label=f'Derivada de {func_name}', color=color_deriv, linewidth=2, linestyle='--')
    plt.title(f'Derivada de {func_name}', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

def main():
    x = np.linspace(-10, 10, 400)

    # Graficar cada función con su derivada en una misma ventana
    plt.figure(figsize=(8, 6))  # Crear una figura para cada par de funciones
    plot_function_and_derivative(x, relu, relu_derivative, 'ReLU', '#1f77b4', '#d62728')
    plt.tight_layout()  # Ajustar el espacio entre subgráficas
    plt.show()

    plt.figure(figsize=(8, 6))
    plot_function_and_derivative(x, sigmoid, sigmoid_derivative, 'Sigmoide', '#2ca02c', '#9467bd')
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 6))
    plot_function_and_derivative(x, tanh, tanh_derivative, 'Tanh', '#ff7f0e', '#8c564b')
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 6))
    plot_function_and_derivative(x, leaky_relu, leaky_relu_derivative, 'Leaky ReLU', '#17becf', '#e377c2')
    plt.tight_layout()
    plt.show()

    # Softmax y Swish sin derivada definida
    plt.figure(figsize=(8, 6))
    plt.plot(x, swish(x), label='Swish', color='#f29e29', linewidth=2)
    plt.title('Swish', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.show()

    x_softmax = np.linspace(-5, 5, 10)
    plt.figure(figsize=(8, 6))
    plt.plot(x_softmax, softmax(x_softmax), label='Softmax', color='#9467bd', linewidth=2)
    plt.title('Softmax', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
