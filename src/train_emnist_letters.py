# import os
# import gzip
# import struct
# import numpy as np

# from model import CNN
# from loss import cross_entropy_loss, cross_entropy_grad


# DATA_DIR = "data/emnist/gzip"

# TRAIN_IMAGES = os.path.join(DATA_DIR, "emnist-letters-train-images-idx3-ubyte.gz")
# TRAIN_LABELS = os.path.join(DATA_DIR, "emnist-letters-train-labels-idx1-ubyte.gz")
# TEST_IMAGES = os.path.join(DATA_DIR, "emnist-letters-test-images-idx3-ubyte.gz")
# TEST_LABELS = os.path.join(DATA_DIR, "emnist-letters-test-labels-idx1-ubyte.gz")


# def load_images(path):
#     with gzip.open(path, "rb") as f:
#         magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
#         data = np.frombuffer(f.read(), dtype=np.uint8)
#         images = data.reshape(num, rows, cols)

#         # изображения нужно повернуть/отразить
#         images = np.transpose(images, (0, 2, 1))
#         images = np.flip(images, axis=2)

#         return images.reshape(num, rows * cols) / 255.0


# def load_labels(path):
#     with gzip.open(path, "rb") as f:
#         magic, num = struct.unpack(">II", f.read(8))
#         labels = np.frombuffer(f.read(), dtype=np.uint8)

#         # в EMNIST Letters метки идут от 1 до 26, делаем 0–25
#         return labels - 1


# print("Загрузка EMNIST Letters...")

# X_train = load_images(TRAIN_IMAGES)
# y_train = load_labels(TRAIN_LABELS)

# X_test = load_images(TEST_IMAGES)
# y_test = load_labels(TEST_LABELS)

# # перемешиваем обучающую выборку
# train_indexes = np.arange(len(X_train))
# np.random.shuffle(train_indexes)

# X_train = X_train[train_indexes]
# y_train = y_train[train_indexes]

# # перемешиваем тестовую выборку
# test_indexes = np.arange(len(X_test))
# np.random.shuffle(test_indexes)

# X_test = X_test[test_indexes]
# y_test = y_test[test_indexes]


# X_train = X_train[:30000]
# y_train = y_train[:30000]

# X_test = X_test[:5000]
# y_test = y_test[:5000]

# model = CNN(num_classes=26)

# epochs = 3
# learning_rate = 0.01

# print("Начинаем обучение на буквах...")

# for epoch in range(epochs):
#     total_loss = 0

#     indexes = np.arange(len(X_train))
#     np.random.shuffle(indexes)

#     for i in indexes:
#         x = X_train[i].reshape(1, -1)
#         target = np.array([y_train[i]])

#         output = model.forward(x)

#         loss = cross_entropy_loss(output, target)
#         total_loss += loss

#         grad = cross_entropy_grad(output, target)
#         model.backward(grad, learning_rate)

#     avg_loss = total_loss / len(X_train)

#     correct = 0
#     for i in range(len(X_test)):
#         x = X_test[i].reshape(1, -1)
#         pred = model.predict(x)[0]

#         if pred == y_test[i]:
#             correct += 1

#     accuracy = correct / len(X_test)

#     # print("Первые 20 правильных меток:", y_test[:20])

#     # test_preds = []
#     # for i in range(20):
#     #     x = X_test[i].reshape(1, -1)
#     #     pred = model.predict(x)[0]
#     #     test_preds.append(pred)

#     # print("Первые 20 предсказаний:", test_preds)

#     print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}")

# model.save("models/emnist_cnn_letters_model.npz")
# print("Модель сохранена в models/emnist_cnn_letters_model.npz")


import os
import gzip
import struct
import numpy as np
import cv2

from model import CNN
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

        images = np.transpose(images, (0, 2, 1))

        return images.reshape(num, rows * cols) / 255.0


def load_labels(path):
    with gzip.open(path, "rb") as f:
        magic, num = struct.unpack(">II", f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
        return labels - 1


def augment_image(flat_image):
    img = flat_image.reshape(28, 28)

    # случайный поворот
    angle = np.random.uniform(-15, 15)
    matrix = cv2.getRotationMatrix2D((14, 14), angle, 1.0)
    img = cv2.warpAffine(img, matrix, (28, 28), borderValue=0)

    # случайный сдвиг
    shift_x = np.random.randint(-3, 4)
    shift_y = np.random.randint(-3, 4)
    matrix = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    img = cv2.warpAffine(img, matrix, (28, 28), borderValue=0)

    # byjulf утолщаем линии
    if np.random.rand() < 0.5:
        kernel = np.ones((2, 2), np.uint8)
        img = cv2.dilate(img, kernel, iterations=1)

    # иногда слегка размываем
    if np.random.rand() < 0.3:
        img = cv2.GaussianBlur(img, (3, 3), 0)

    img = np.clip(img, 0, 1)

    return img.reshape(784)


def check_no_overlap(X1, X2, name1, name2):
    set1 = set(row.tobytes() for row in (X1 * 255).astype(np.uint8))
    set2 = set(row.tobytes() for row in (X2 * 255).astype(np.uint8))

    overlap = len(set1.intersection(set2))

    print(f"Пересечение {name1} и {name2}: {overlap}")

    if overlap > 0:
        raise ValueError(f"Ошибка: найдено пересечение между {name1} и {name2}")


print("Загрузка EMNIST Letters...")

X_train = load_images(TRAIN_IMAGES)
y_train = load_labels(TRAIN_LABELS)

X_test = load_images(TEST_IMAGES)
y_test = load_labels(TEST_LABELS)


# перемешиваем обучающую выборку
train_indexes = np.arange(len(X_train))
np.random.shuffle(train_indexes)

X_train = X_train[train_indexes]
y_train = y_train[train_indexes]


# делим train на train и validation
validation_size = int(len(X_train) * 0.15)

X_val = X_train[:validation_size]
y_val = y_train[:validation_size]

X_train = X_train[validation_size:]
y_train = y_train[validation_size:]


# перемешиваем тестовую выборку
test_indexes = np.arange(len(X_test))
np.random.shuffle(test_indexes)

X_test = X_test[test_indexes]
y_test = y_test[test_indexes]


# ограничиваем объём данных, чтобы обучение на NumPy не шло слишком долго
X_train = X_train[:30000]
y_train = y_train[:30000]

X_val = X_val[:5000]
y_val = y_val[:5000]

X_test = X_test[:5000]
y_test = y_test[:5000]


# проверяем, что используемые данные не пересекаются
check_no_overlap(X_train, X_val, "train", "validation")
check_no_overlap(X_train, X_test, "train", "test")
check_no_overlap(X_val, X_test, "validation", "test")


model = CNN(num_classes=26)

epochs = 5
learning_rate = 0.01

print("Начинаем обучение CNN на буквах...")

for epoch in range(epochs):
    total_loss = 0

    indexes = np.arange(len(X_train))
    np.random.shuffle(indexes)

    for i in indexes:
        # аугментация применяется только к обучающей выборке
        x_aug = augment_image(X_train[i])
        x = x_aug.reshape(1, -1)

        target = np.array([y_train[i]])

        output = model.forward(x)

        loss = cross_entropy_loss(output, target)
        total_loss += loss

        grad = cross_entropy_grad(output, target)
        model.backward(grad, learning_rate)

    avg_loss = total_loss / len(X_train)

    # validation accuracy считаем без аугментации
    correct = 0

    for i in range(len(X_val)):
        x = X_val[i].reshape(1, -1)
        pred = model.predict(x)[0]

        if pred == y_val[i]:
            correct += 1

    val_accuracy = correct / len(X_val)

    print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}, Val Accuracy: {val_accuracy:.2f}")


# финальная проверка на test
correct = 0

for i in range(len(X_test)):
    x = X_test[i].reshape(1, -1)
    pred = model.predict(x)[0]

    if pred == y_test[i]:
        correct += 1

test_accuracy = correct / len(X_test)

print(f"Final Test Accuracy: {test_accuracy:.2f}")

model.save("models/emnist_cnn_letters_model.npz")
print("Модель сохранена в models/emnist_cnn_letters_model.npz")