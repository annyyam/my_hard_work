import numpy as np

class ReLULayer:
    def __init__(self):
        self.input = None

    def forward(self, input):
        self.input = input
        return np.maximum(0, input)

    def backward(self, grad_output):
        return grad_output * (self.input > 0)
    

class DenseLayer:
    def __init__(self, input_size, output_size):
        # веса и смещения
        self.weights = np.random.randn(input_size, output_size) * 0.01
        self.bias = np.zeros((1, output_size))
        
        self.input = None

    def forward(self, input):
        self.input = input
        return np.dot(input, self.weights) + self.bias

    def backward(self, grad_output, learning_rate):
        # градиенты
        grad_weights = np.dot(self.input.T, grad_output)
        grad_bias = np.sum(grad_output, axis=0, keepdims=True)
        
        # градиент для предыдущего слоя
        grad_input = np.dot(grad_output, self.weights.T)
        
        # обновление весов
        self.weights -= learning_rate * grad_weights
        self.bias -= learning_rate * grad_bias
        
        return grad_input    
    

class SoftmaxLayer:
    def forward(self, input):
        exps = np.exp(input - np.max(input, axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)

    def backward(self, grad_output):
        return grad_output    