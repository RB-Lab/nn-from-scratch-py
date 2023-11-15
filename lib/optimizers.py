import numpy as np

# SGD optimizer
class Optimizer_SGD():
    def __init__(self, learning_rate=1.0, decay=0.0, momentum=0.0):
        self.learning_rate = learning_rate
        self.decay = decay
        self.momentum = momentum
    
    def get_lr(self, epoch):
        return self.learning_rate * (1. / (1. + self.decay * epoch))

    def update_params(self, layer, epoch):
        if(not hasattr(layer, 'w_moment')):
            layer.w_moment = np.zeros_like(layer.weights)
            layer.b_moment = np.zeros_like(layer.biases)

        if(self.momentum):
            weights_updates = self.momentum * layer.w_moment - self.get_lr(epoch) * layer.d_weights
            layer.w_moment = weights_updates

            bises_updates = self.momentum * layer.b_moment - self.get_lr(epoch) * layer.d_biases
            layer.b_moment = bises_updates
        else:
            weights_updates = -self.get_lr(epoch) * layer.d_weights
            bises_updates = -self.get_lr(epoch) * layer.d_biases

        layer.weights += weights_updates
        layer.biases += bises_updates
    
class Optimizer_RMSProp:
    def __init__(self, learning_rate=0.001, decay=0.0, epsilon=1e-7, rho=0.9):
        self.learning_rate = learning_rate
        self.decay = decay
        self.epsilon = epsilon
        self.rho = rho

    def get_lr(self, epoch):
        return self.learning_rate * (1. / (1. + self.decay * epoch))

    def update_params(self, layer, epoch):
        if(not hasattr(layer, 'w_cache')):
            layer.w_cache = np.zeros_like(layer.weights)
            layer.b_cache = np.zeros_like(layer.biases)

        layer.w_cache = self.rho * layer.w_cache + (1 - self.rho) * layer.d_weights**2
        layer.b_cache = self.rho * layer.b_cache + (1 - self.rho) * layer.d_biases**2

        layer.weights += -self.get_lr(epoch) * layer.d_weights / (np.sqrt(layer.w_cache) + self.epsilon)
        layer.biases += -self.get_lr(epoch) * layer.d_biases / (np.sqrt(layer.b_cache) + self.epsilon)

class Optimizer_Adam:
    def __init__(self, learning_rate=0.001, decay=0.0, epsilon=1e-7, beta_1=0.9, beta_2=0.999):
        self.learning_rate = learning_rate
        self.decay = decay
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2
    
    def get_lr(self, epoch):
        return self.learning_rate * (1. / (1. + self.decay * epoch))
    
    def update_params(self, layer, epoch):
        if(not hasattr(layer, 'w_cache')):
            layer.w_cache = np.zeros_like(layer.weights)
            layer.b_cache = np.zeros_like(layer.biases)
            layer.w_moment = np.zeros_like(layer.weights)
            layer.b_moment = np.zeros_like(layer.biases)

        layer.w_moment = self.beta_1 * layer.w_moment + (1 - self.beta_1) * layer.d_weights
        layer.b_moment = self.beta_1 * layer.b_moment + (1 - self.beta_1) * layer.d_biases

        w_moment_corrected = layer.w_moment / (1 - self.beta_1 ** (epoch + 1))
        b_moment_corrected = layer.b_moment / (1 - self.beta_1 ** (epoch + 1))

        layer.w_cache = self.beta_2 * layer.w_cache + (1 - self.beta_2) * layer.d_weights**2
        layer.b_cache = self.beta_2 * layer.b_cache + (1 - self.beta_2) * layer.d_biases**2

        w_cache_corrected = layer.w_cache / (1 - self.beta_2 ** (epoch + 1))
        b_cache_corrected = layer.b_cache / (1 - self.beta_2 ** (epoch + 1))

        layer.weights += -self.get_lr(epoch) * w_moment_corrected / (np.sqrt(w_cache_corrected) + self.epsilon)
        layer.biases += -self.get_lr(epoch) * b_moment_corrected / (np.sqrt(b_cache_corrected) + self.epsilon)
        