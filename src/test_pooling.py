import numpy as np
from layers import MaxPoolingLayer

pool = MaxPoolingLayer(pool_size=2, stride=2)

x = np.random.rand(1, 2, 26, 26)

out = pool.forward(x)

print("Input shape:", x.shape)
print("Output shape:", out.shape)