import numpy as np
  # ReLU activation
class Activation_ReLU:
    #  Forward pass
    def forward(self, inputs):
        self.inputs = inputs
        # Calculate output values from inputs 
        self.output = np.maximum(0, inputs)
    
    def backward(self, d_values):
        # Since we need to modify original variable, let's make a copy of values first
        self.d_inputs = d_values.copy()
        # Zero gradient where input values were negative
        self.d_inputs[self.inputs <= 0] = 0

# Softmax activation
class Activation_Softmax: 
    # Forward pass
    def forward(self, inputs):
        # Get un-normalized probabilities
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims= True)) 
        probabilities = exp_values / np.sum(exp_values, axis=1,keepdims = True)
        # Normalize them for each sample 
        self.output = probabilities
    
    def backward(self, d_values):
        """
        d_values is the matrix where each row contains partial derivatives of a loss function 
        with respect to each softmax output (which is an array of probabilities for each class)
        """
        # Create uninitialized array
        self.d_inputs = np.empty_like(d_values)
        # Enumerate outputs and gradients
        for index, (single_output, single_d_values) in enumerate(zip(self.output, d_values)):
            # Flatten output array
            single_output = single_output.reshape(-1, 1)
            # Calculate Jacobian matrix of the output
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            # Calculate sample-wise gradient and add it to the array of sample gradients 
            self.d_inputs[index] = np.dot(jacobian_matrix, single_d_values)
