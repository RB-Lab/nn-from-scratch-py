import numpy as np
# Dense layer
class Layer_Dense:
    # Layer initialization
    def __init__(self, n_inputs, n_neurons):
        # Initialize weights and biases
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons) 
        self.biases = np.zeros((1, n_neurons))
    # Forward pass
    def forward(self, inputs):
        self.inputs = inputs
        # Calculate output values from inputs, weights and biases 
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, d_values):
        # Gradients on parameters
        self.d_weights = np.dot(self.inputs.T, d_values)
        self.d_biases = np.sum(d_values, axis=0, keepdims=True)
        # Gradient on values
        self.d_inputs = np.dot(d_values, self.weights.T)