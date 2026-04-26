import numpy as np
from layers import ReLULayer

relu = ReLULayer()

x = np.array([-2, -1, 0, 1, 2])
print("Input:", x)

out = relu.forward(x)
print("Forward:", out)

grad = np.ones_like(x)
back = relu.backward(grad)
print("Backward:", back)