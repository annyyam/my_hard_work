import numpy as np
from model import NeuralNetwork
from loss import cross_entropy_loss, cross_entropy_grad

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


print("Загрузка данных...")

digits = load_digits()

X = digits.data / 16.0
y = digits.target.astype(int)

# приводим 8x8 к формату 784, чтобы подходило под нашу сеть
X_padded = np.zeros((X.shape[0], 784))

for i in range(X.shape[0]):
    image_8x8 = X[i].reshape(8, 8)
    image_28x28 = np.zeros((28, 28))

    # вставляем маленькую цифру в центр 28x28
    image_28x28[10:18, 10:18] = image_8x8

    X_padded[i] = image_28x28.reshape(-1)

X = X_padded

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = NeuralNetwork()

epochs = 20
learning_rate = 0.1

print("Начинаем обучение...")

for epoch in range(epochs):
    total_loss = 0

    for i in range(len(X_train)):
        x = X_train[i].reshape(1, -1)
        target = np.array([y_train[i]])

        output = model.forward(x)

        loss = cross_entropy_loss(output, target)
        total_loss += loss

        grad = cross_entropy_grad(output, target)
        model.backward(grad, learning_rate)

    avg_loss = total_loss / len(X_train)
    print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}")


correct = 0

for i in range(len(X_test)):
    x = X_test[i].reshape(1, -1)
    pred = model.predict(x)[0]

    if pred == y_test[i]:
        correct += 1

accuracy = correct / len(X_test)
print(f"Accuracy: {accuracy:.2f}")

model.save("models/digits_model.npz")
print("Модель сохранена в models/digits_model.npz")