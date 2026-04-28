import os
import gzip
import urllib.request
import struct
import numpy as np

from model import NeuralNetwork
from loss import cross_entropy_loss, cross_entropy_grad


BASE_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/"
DATA_DIR = "data/mnist"

FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}


def download_file(filename):
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, filename)

    if not os.path.exists(path):
        print(f"Скачиваю {filename}...")
        urllib.request.urlretrieve(BASE_URL + filename, path)

    return path


def load_images(path):
    with gzip.open(path, "rb") as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        data = np.frombuffer(f.read(), dtype=np.uint8)
        return data.reshape(num, rows * cols) / 255.0


def load_labels(path):
    with gzip.open(path, "rb") as f:
        magic, num = struct.unpack(">II", f.read(8))
        return np.frombuffer(f.read(), dtype=np.uint8)


print("Загрузка MNIST...")

train_images_path = download_file(FILES["train_images"])
train_labels_path = download_file(FILES["train_labels"])
test_images_path = download_file(FILES["test_images"])
test_labels_path = download_file(FILES["test_labels"])

X_train = load_images(train_images_path)
y_train = load_labels(train_labels_path)

X_test = load_images(test_images_path)
y_test = load_labels(test_labels_path)

# ограничиваем размер, чтобы NumPy-обучение не шло слишком долго
X_train = X_train[:10000]
y_train = y_train[:10000]

X_test = X_test[:2000]
y_test = y_test[:2000]

model = NeuralNetwork()

epochs = 5
learning_rate = 0.05

print("Начинаем обучение на MNIST...")

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

model.save("models/mnist_model.npz")
print("Модель сохранена в models/mnist_model.npz")