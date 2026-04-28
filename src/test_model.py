import numpy as np
from model import NeuralNetwork

model = NeuralNetwork()

x = np.random.rand(1, 784)

out = model.forward(x)

print("Output:", out)
print("Shape:", out.shape)
print("Sum:", np.sum(out))
print("Prediction:", model.predict(x))