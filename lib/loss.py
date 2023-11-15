import numpy as np

# Common loss class


class Loss:
    # Calculates the data and regularization losses
    # given model output and ground truth values
    def calculate(self, output, y):
        """
        Arguments:
            y_pred: array of vectors, each containing probabilities of a sample belonging to a particular class.
            y_true: vector of labels, or array of one-hot encoded vectors, correct values.
        Returns:
            float, data loss.
        """
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)

        return data_loss
    
    # Calculates the regularization loss
    def regularization_loss(self, layer):
        regularization_loss = 0
        # L1 regularization - weights
        # calculate only when factor greater than 0
        if layer.weight_regularizer_l1 > 0:
            regularization_loss += layer.weight_regularizer_l1 * np.sum(np.abs(layer.weights))
        # L2 regularization - weights
        if layer.weight_regularizer_l2 > 0:
            regularization_loss += layer.weight_regularizer_l2 * np.sum(layer.weights * layer.weights)
        # L1 regularization - biases
        # calculate only when factor greater than 0
        if layer.bias_regularizer_l1 > 0:
            regularization_loss += layer.bias_regularizer_l1 * np.sum(np.abs(layer.biases))
        # L2 regularization - biases
        if layer.bias_regularizer_l2 > 0:
            regularization_loss += layer.bias_regularizer_l2 * np.sum(layer.biases * layer.biases)
        return regularization_loss

# Cross-entropy loss


class Loss_CategoricalCrossentropy(Loss):  # Forward pass
    def forward(self, y_pred, y_true):
        """
        Arguments:
            y_pred: array of vectors, each containing probabilities of a sample belonging to a particular class.
            y_true: vector of labels, or array of one-hot encoded vectors, correct values.
        """
        # Number of samples in a batch
        samples = len(y_pred)
        # Clip data to prevent division by 0
        # Clip both sides to not drag mean towards any value
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        # Probabilities for target values - # only if categorical labels
        # correct_confidences – array of predicted probabilities for correct labels (for each sample).
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[
                range(samples), # for each sample
                y_true # get value from y_pred_clipped at label's index
            ]
        # Mask values - only for one-hot encoded labels
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(
                y_pred_clipped * y_true, # multiply each row of y_pred_clipped by corresponding row of y_true
                axis=1
            )
        # Losses
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

    def backward(self, d_values, y_true):
        # Number of samples
        samples = len(d_values)
        # Number of labels in every sample
        # We'll use the first sample to count them
        labels = len(d_values[0])
        # If labels are sparse, turn them into one-hot vector
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]
        # Calculate gradient
        self.d_inputs = -y_true / d_values
        # Normalize gradient
        self.d_inputs = self.d_inputs / samples
