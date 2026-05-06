import os
import gzip
import struct
import numpy as np
import matplotlib.pyplot as plt

from model import CNN


DATA_DIR = "data/emnist/gzip"

TEST_IMAGES = os.path.join(DATA_DIR, "emnist-letters-test-images-idx3-ubyte.gz")
TEST_LABELS = os.path.join(DATA_DIR, "emnist-letters-test-labels-idx1-ubyte.gz")


def load_images(path):
    with gzip.open(path, "rb") as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        data = np.frombuffer(f.read(), dtype=np.uint8)
        images = data.reshape(num, rows, cols)

        # та же ориентвция, что и при обучении
        images = np.transpose(images, (0, 2, 1))
        images = np.flip(images, axis=1)

        return images.reshape(num, rows * cols) / 255.0


def load_labels(path):
    with gzip.open(path, "rb") as f:
        magic, num = struct.unpack(">II", f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)

        return labels - 1


def letter_to_index(letter):
    return ord(letter.upper()) - ord("A")


def index_to_letter(index):
    return chr(index + ord("A"))


X_test = load_images(TEST_IMAGES)
y_test = load_labels(TEST_LABELS)

model = CNN(num_classes=26)
model.load("models/emnist_cnn_letters_model.npz")

letters_to_check = ["K", "V", "M", "A"]

plot_number = 1

plt.figure(figsize=(10, 8))

for letter in letters_to_check:
    target_index = letter_to_index(letter)
    indexes = np.where(y_test == target_index)[0][:5]

    for image_index in indexes:
        x = X_test[image_index].reshape(1, -1)

        probs = model.forward(x)[0]
        pred = np.argmax(probs)

        top_indexes = probs.argsort()[-3:][::-1]
        top_text = ", ".join(
            f"{index_to_letter(i)} {probs[i] * 100:.1f}%"
            for i in top_indexes
        )

        print(f"Истина: {letter}, предсказание: {index_to_letter(pred)}, Top-3: {top_text}")

        plt.subplot(len(letters_to_check), 5, plot_number)
        plt.imshow(X_test[image_index].reshape(28, 28), cmap="gray")
        plt.title(f"{letter}->{index_to_letter(pred)}")
        plt.axis("off")

        plot_number += 1

plt.tight_layout()
plt.show()