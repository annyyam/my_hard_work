import numpy as np
from layers import DenseLayer, ReLULayer, SoftmaxLayer


class NeuralNetwork:
    def __init__(self, input_size=784, num_classes=10):
        self.num_classes = num_classes

        self.layers = [
            DenseLayer(input_size, 128),
            ReLULayer(),
            DenseLayer(128, 64),
            ReLULayer(),
            DenseLayer(64, num_classes),
            SoftmaxLayer()
        ]

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad, learning_rate):
        for layer in reversed(self.layers):
            if isinstance(layer, DenseLayer):
                grad = layer.backward(grad, learning_rate)
            else:
                grad = layer.backward(grad)

    def predict(self, x):
        probabilities = self.forward(x)
        return np.argmax(probabilities, axis=1)

    def save(self, path):
        weights = {}
        index = 0

        for layer in self.layers:
            if isinstance(layer, DenseLayer):
                weights[f"w{index}"] = layer.weights
                weights[f"b{index}"] = layer.bias
                index += 1

        np.savez(path, **weights)

    def load(self, path):
        data = np.load(path)
        index = 0

        for layer in self.layers:
            if isinstance(layer, DenseLayer):
                layer.weights = data[f"w{index}"]
                layer.bias = data[f"b{index}"]
                index += 1


# import numpy as np
# from layers import DenseLayer, ReLULayer, SoftmaxLayer


# class NeuralNetwork:
#     def __init__(self):
#         self.layers = [
#             DenseLayer(784, 128),
#             ReLULayer(),
#             DenseLayer(128, 64),
#             ReLULayer(),
#             DenseLayer(64, 10),
#             SoftmaxLayer()
#         ]

#     def forward(self, x):
#         for layer in self.layers:
#             x = layer.forward(x)
#         return x

#     def backward(self, grad, learning_rate):
#         for layer in reversed(self.layers):
#             if isinstance(layer, DenseLayer):
#                 grad = layer.backward(grad, learning_rate)
#             else:
#                 grad = layer.backward(grad)

#     def predict(self, x):
#         probabilities = self.forward(x)
#         return np.argmax(probabilities, axis=1)

#     def save(self, path):
#         weights = {}
#         index = 0

#         for layer in self.layers:
#             if isinstance(layer, DenseLayer):
#                 weights[f"w{index}"] = layer.weights
#                 weights[f"b{index}"] = layer.bias
#                 index += 1

#         np.savez(path, **weights)

#     def load(self, path):
#         data = np.load(path)
#         index = 0

#         for layer in self.layers:
#             if isinstance(layer, DenseLayer):
#                 layer.weights = data[f"w{index}"]
#                 layer.bias = data[f"b{index}"]
#                 index += 1