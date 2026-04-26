import numpy as np
from layers import DenseLayer

np.random.seed(0)

dense = DenseLayer(3, 2)

x = np.array([[1, 2, 3]])
print("Input:", x)

out = dense.forward(x)
print("Forward:", out)

grad = np.ones_like(out)
back = dense.backward(grad, learning_rate=0.01)

print("Backward:", back)