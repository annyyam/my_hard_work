import os
import gzip
import struct
import numpy as np
import matplotlib.pyplot as plt


DATA_DIR = "data/emnist/gzip"

TRAIN_IMAGES = os.path.join(DATA_DIR, "emnist-letters-train-images-idx3-ubyte.gz")
TRAIN_LABELS = os.path.join(DATA_DIR, "emnist-letters-train-labels-idx1-ubyte.gz")


def load_images(path):
    with gzip.open(path, "rb") as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        data = np.frombuffer(f.read(), dtype=np.uint8)
        images = data.reshape(num, rows, cols)
        
        images = np.transpose(images, (0, 2, 1))
        images = np.flip(images, axis=1)

        return images / 255.0


def load_labels(path):
    with gzip.open(path, "rb") as f:
        magic, num = struct.unpack(">II", f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
        return labels - 1


images = load_images(TRAIN_IMAGES)
labels = load_labels(TRAIN_LABELS)

target_letter = "K"
target_index = ord(target_letter) - ord("A")

indexes = np.where(labels == target_index)[0][:12]

for plot_index, image_index in enumerate(indexes):
    plt.subplot(3, 4, plot_index + 1)
    plt.imshow(images[image_index], cmap="gray")
    plt.title(target_letter)
    plt.axis("off")

plt.show()