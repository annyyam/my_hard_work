import os
import gzip
import struct
import numpy as np

from model import NeuralNetwork
from loss import cross_entropy_loss, cross_entropy_grad


DATA_DIR = "data/emnist/gzip"

TRAIN_IMAGES = os.path.join(DATA_DIR, "emnist-letters-train-images-idx3-ubyte.gz")
TRAIN_LABELS = os.path.join(DATA_DIR, "emnist-letters-train-labels-idx1-ubyte.gz")
TEST_IMAGES = os.path.join(DATA_DIR, "emnist-letters-test-images-idx3-ubyte.gz")
TEST_LABELS = os.path.join(DATA_DIR, "emnist-letters-test-labels-idx1-ubyte.gz")


def load_images(path):
    with gzip.open(path, "rb") as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        data = np.frombuffer(f.read(), dtype=np.uint8)
        images = data.reshape(num, rows, cols)

        # изображения нужно повернуть/отразить
        images = np.transpose(images, (0, 2, 1))
        images = np.flip(images, axis=2)

        return images.reshape(num, rows * cols) / 255.0


def load_labels(path):
    with gzip.open(path, "rb") as f:
        magic, num = struct.unpack(">II", f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)

        # в EMNIST Letters метки идут от 1 до 26, делаем 0–25
        return labels - 1


print("Загрузка EMNIST Letters...")

X_train = load_images(TRAIN_IMAGES)
y_train = load_labels(TRAIN_LABELS)

X_test = load_images(TEST_IMAGES)
y_test = load_labels(TEST_LABELS)

# X_train = X_train[:15000]
# y_train = y_train[:15000]

# X_test = X_test[:3000]
# y_test = y_test[:3000]

model = NeuralNetwork(num_classes=26)

epochs = 10
learning_rate = 0.02

print("Начинаем обучение на буквах...")

for epoch in range(epochs):
    total_loss = 0

    indexes = np.arange(len(X_train))
    np.random.shuffle(indexes)

    for i in indexes:
        x = X_train[i].reshape(1, -1)
        target = np.array([y_train[i]])

        output = model.forward(x)

        loss = cross_entropy_loss(output, target)
        total_loss += loss

        grad = cross_entropy_grad(output, target)
        model.backward(grad, learning_rate)

    avg_loss = total_loss / len(X_train)

    correct = 0
    for i in range(len(X_test)):
        x = X_test[i].reshape(1, -1)
        pred = model.predict(x)[0]

        if pred == y_test[i]:
            correct += 1

    accuracy = correct / len(X_test)

    print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}")


model.save("models/emnist_letters_model.npz")
print("Модель сохранена в models/emnist_letters_model.npz")