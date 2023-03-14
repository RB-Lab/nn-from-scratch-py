import numpy as np
from lib.activations import Activation_Softmax
from lib.loss import Loss_CategoricalCrossentropy

# Softmax activation + Categorical Cross-entropy loss combined into a single class
class Activation_Softmax_Loss_CategoricalCrossentropy():
    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss = Loss_CategoricalCrossentropy()

    def forward(self, inputs, y_true):
        self.activation.forward(inputs)
        self.output = self.activation.output
        return self.loss.calculate(self.output, y_true)

    def backward(self, d_values, y_true):
        samples = len(d_values)
        # If labels are one-hot encoded, turn them into discrete values
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
        self.d_inputs = d_values.copy()
        # Calculate gradient
        self.d_inputs[range(samples), y_true] -= 1
        # Normalize gradient
        self.d_inputs = self.d_inputs / samples