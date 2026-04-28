import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from model import NeuralNetwork


digits = load_digits()

X = digits.data / 16.0
y = digits.target.astype(int)

# берём один пример
index = 0

image_8x8 = X[index].reshape(8, 8)

image_28x28 = np.zeros((28, 28))
image_28x28[10:18, 10:18] = image_8x8

x = image_28x28.reshape(1, 784)

model = NeuralNetwork()
model.load("models/digits_model.npz")

prediction = model.predict(x)[0]

print("Правильный ответ:", y[index])
print("Предсказание модели:", prediction)

plt.imshow(image_28x28, cmap="gray")
plt.title(f"True: {y[index]}, Predicted: {prediction}")
plt.show()