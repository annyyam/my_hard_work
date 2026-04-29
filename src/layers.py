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
    
class ConvLayer:
    def __init__(self, in_channels, out_channels, kernel_size=3, learning_rate=0.01):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.learning_rate = learning_rate

        self.weights = np.random.randn(
            out_channels,
            in_channels,
            kernel_size,
            kernel_size
        ) * np.sqrt(2.0 / (in_channels * kernel_size * kernel_size))

        self.bias = np.zeros(out_channels)

        self.input = None

    def forward(self, input):
        self.input = input

        batch_size, in_channels, height, width = input.shape
        output_height = height - self.kernel_size + 1
        output_width = width - self.kernel_size + 1

        output = np.zeros((batch_size, self.out_channels, output_height, output_width))

        for b in range(batch_size):
            for f in range(self.out_channels):
                for i in range(output_height):
                    for j in range(output_width):
                        region = input[
                            b,
                            :,
                            i:i + self.kernel_size,
                            j:j + self.kernel_size
                        ]

                        output[b, f, i, j] = np.sum(region * self.weights[f]) + self.bias[f]

        return output

    def backward(self, grad_output):
        batch_size, in_channels, height, width = self.input.shape
        _, out_channels, output_height, output_width = grad_output.shape

        grad_input = np.zeros_like(self.input)
        grad_weights = np.zeros_like(self.weights)
        grad_bias = np.zeros_like(self.bias)

        for b in range(batch_size):
            for f in range(out_channels):
                for i in range(output_height):
                    for j in range(output_width):
                        region = self.input[
                            b,
                            :,
                            i:i + self.kernel_size,
                            j:j + self.kernel_size
                        ]

                        grad_weights[f] += grad_output[b, f, i, j] * region
                        grad_input[
                            b,
                            :,
                            i:i + self.kernel_size,
                            j:j + self.kernel_size
                        ] += grad_output[b, f, i, j] * self.weights[f]
                        grad_bias[f] += grad_output[b, f, i, j]

        self.weights -= self.learning_rate * grad_weights
        self.bias -= self.learning_rate * grad_bias

        return grad_input    