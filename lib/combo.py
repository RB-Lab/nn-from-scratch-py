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

    def backward(self, y_pred, y_true):
        samples = len(y_pred)
        # If labels are one-hot encoded, turn them into discrete values
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
        # the derivative of cross entropy and softmax combination with the respect to softmax input
        # is y_predᵢₖ - y_trueᵢₖ, where y_trueᵢₖ is one-hot encoded, it basically
        # means that if y_trueᵢₖ == 1, the derivative is y_predᵢₖ - 1 and just y_predᵢₖ when y_trueᵢₖ == 0
        # so we can just copy y_pred to d_inputs and subtract 1 from the correct labels
        self.d_inputs = y_pred.copy()
        # in each row at index of correct label subtract 1
        self.d_inputs[range(samples), y_true] -= 1
        # Normalize gradient by number of samples in a batch
        self.d_inputs = self.d_inputs / samples