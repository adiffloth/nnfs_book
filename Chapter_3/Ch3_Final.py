# Add dense layer class

import numpy as np
import nnfs
from nnfs.datasets import spiral_data
# import matplotlib.pyplot as plt

nnfs.init()


# Dense layer
class Layer_Dense:

    # Layer initialization
    def __init__(self, n_inputs, n_neurons):
        # Initialize weights and biases
        print('Initializing dense layer object.')
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)  # Start the weights out random and small.
        self.biases = np.zeros((1, n_neurons))  # Common practice to start with zero biases.
        self.output = np.array(0)

    # Forward pass
    def forward(self, inputs):
        # Calculate output values from inputs, weights and biases
        print('Performing forward pass.')
        self.output = np.dot(inputs, self.weights) + self.biases


# Create dataset
X, y = spiral_data(samples=100, classes=3)
print(f'{X.shape=}')
print(f'{y.shape=}')
# plt.scatter(X[:, 0], X[:, 1], c=y, cmap='brg')
# plt.show()

# Create Dense layer with 2 input features and 3 output values
dense1 = Layer_Dense(2, 3)
print(f'{dense1.weights.shape=}')
print(f'{dense1.biases.shape=}')
print(f'{dense1.output.shape=}')

# Perform a forward pass of our training data through this layer
dense1.forward(X)

# Let's see output of the first few samples:
print(f'{dense1.output.shape=}')
print(dense1.output[:5])


'''
>>>
[[ 0.0000000e+00  0.0000000e+00  0.0000000e+00]
 [-1.0475188e-04  1.1395361e-04 -4.7983500e-05]
 [-2.7414842e-04  3.1729150e-04 -8.6921798e-05]
 [-4.2188365e-04  5.2666257e-04 -5.5912682e-05]
 [-5.7707680e-04  7.1401405e-04 -8.9430439e-05]]
'''
