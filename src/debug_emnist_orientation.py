import os
import gzip
import struct
import numpy as np
import matplotlib.pyplot as plt


DATA_DIR = "data/emnist/gzip"

TRAIN_IMAGES = os.path.join(DATA_DIR, "emnist-letters-train-images-idx3-ubyte.gz")
TRAIN_LABELS = os.path.join(DATA_DIR, "emnist-letters-train-labels-idx1-ubyte.gz")


def load_raw_images(path):
    with gzip.open(path, "rb") as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        data = np.frombuffer(f.read(), dtype=np.uint8)
        return data.reshape(num, rows, cols) / 255.0


def load_labels(path):
    with gzip.open(path, "rb") as f:
        magic, num = struct.unpack(">II", f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
        return labels - 1


images = load_raw_images(TRAIN_IMAGES)
labels = load_labels(TRAIN_LABELS)

target_letter = "A"
target_index = ord(target_letter) - ord("A")

image_index = np.where(labels == target_index)[0][0]
img = images[image_index]

variants = {
    "raw": img,
    "transpose only": img.T,
    "transpose + flip left/right": np.fliplr(img.T),
    "transpose + flip up/down": np.flipud(img.T),
    "rot90": np.rot90(img),
    "rot90 + fliplr": np.fliplr(np.rot90(img)),
    "rot90 + flipud": np.flipud(np.rot90(img)),
}

plt.figure(figsize=(12, 6))

for i, (name, variant) in enumerate(variants.items()):
    plt.subplot(2, 4, i + 1)
    plt.imshow(variant, cmap="gray")
    plt.title(name)
    plt.axis("off")

plt.tight_layout()
plt.show()