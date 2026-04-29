import numpy as np
from layers import DenseLayer, ReLULayer, SoftmaxLayer, ConvLayer, MaxPoolingLayer


class CNN:
    def __init__(self, num_classes=26):
        self.num_classes = num_classes

        self.conv1 = ConvLayer(in_channels=1, out_channels=8, kernel_size=3)
        self.relu1 = ReLULayer()
        self.pool1 = MaxPoolingLayer()

        self.conv2 = ConvLayer(in_channels=8, out_channels=16, kernel_size=3)
        self.relu2 = ReLULayer()
        self.pool2 = MaxPoolingLayer()

        # после двух свёрток и pooling размер примерно 5x5
        self.flatten_size = 16 * 5 * 5

        self.fc1 = DenseLayer(self.flatten_size, 128)
        self.relu3 = ReLULayer()
        self.fc2 = DenseLayer(128, num_classes)
        self.softmax = SoftmaxLayer()

    def forward(self, x):
        x = x.reshape(-1, 1, 28, 28)

        x = self.conv1.forward(x)
        x = self.relu1.forward(x)
        x = self.pool1.forward(x)

        x = self.conv2.forward(x)
        x = self.relu2.forward(x)
        x = self.pool2.forward(x)

        x = x.reshape(x.shape[0], -1)

        x = self.fc1.forward(x)
        x = self.relu3.forward(x)
        x = self.fc2.forward(x)
        x = self.softmax.forward(x)

        return x

    def backward(self, grad, learning_rate):
        grad = self.softmax.backward(grad)

        grad = self.fc2.backward(grad, learning_rate)
        grad = self.relu3.backward(grad)
        grad = self.fc1.backward(grad, learning_rate)

        grad = grad.reshape(-1, 16, 5, 5)

        grad = self.pool2.backward(grad)
        grad = self.relu2.backward(grad)
        grad = self.conv2.backward(grad)

        grad = self.pool1.backward(grad)
        grad = self.relu1.backward(grad)
        grad = self.conv1.backward(grad)

    def predict(self, x):
        probs = self.forward(x)
        return np.argmax(probs, axis=1)


# import numpy as np
# from layers import DenseLayer, ReLULayer, SoftmaxLayer


# class NeuralNetwork:
#     def __init__(self, input_size=784, num_classes=10):
#         self.num_classes = num_classes

#         self.layers = [
#             DenseLayer(input_size, 128),
#             ReLULayer(),
#             DenseLayer(128, 64),
#             ReLULayer(),
#             DenseLayer(64, num_classes),
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