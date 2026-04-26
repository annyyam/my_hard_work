import numpy as np

class ReLULayer:
    def __init__(self):
        self.input = None

    def forward(self, input):
        self.input = input
        return np.maximum(0, input)

    def backward(self, grad_output):
        return grad_output * (self.input > 0)