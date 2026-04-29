import numpy as np
from model import CNN

model = CNN(num_classes=26)

x = np.random.rand(1, 784)

out = model.forward(x)

print("Output shape:", out.shape)
print("Sum:", out.sum())