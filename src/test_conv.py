import numpy as np
from layers import ConvLayer

conv = ConvLayer(in_channels=1, out_channels=2, kernel_size=3)

x = np.random.rand(1, 1, 28, 28)

out = conv.forward(x)

print("Input shape:", x.shape)
print("Output shape:", out.shape)