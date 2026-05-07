# # import os
# # import gzip
# # import struct
# # import numpy as np

# # from model import CNN
# # from loss import cross_entropy_loss, cross_entropy_grad


# # DATA_DIR = "data/emnist/gzip"

# # TRAIN_IMAGES = os.path.join(DATA_DIR, "emnist-letters-train-images-idx3-ubyte.gz")
# # TRAIN_LABELS = os.path.join(DATA_DIR, "emnist-letters-train-labels-idx1-ubyte.gz")
# # TEST_IMAGES = os.path.join(DATA_DIR, "emnist-letters-test-images-idx3-ubyte.gz")
# # TEST_LABELS = os.path.join(DATA_DIR, "emnist-letters-test-labels-idx1-ubyte.gz")


# # def load_images(path):
# #     with gzip.open(path, "rb") as f:
# #         magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
# #         data = np.frombuffer(f.read(), dtype=np.uint8)
# #         images = data.reshape(num, rows, cols)

# #         # изображения нужно повернуть/отразить
# #         images = np.transpose(images, (0, 2, 1))
# #         images = np.flip(images, axis=2)

# #         return images.reshape(num, rows * cols) / 255.0


# # def load_labels(path):
# #     with gzip.open(path, "rb") as f:
# #         magic, num = struct.unpack(">II", f.read(8))
# #         labels = np.frombuffer(f.read(), dtype=np.uint8)

# #         # в EMNIST Letters метки идут от 1 до 26, делаем 0–25
# #         return labels - 1


# # print("Загрузка EMNIST Letters...")

# # X_train = load_images(TRAIN_IMAGES)
# # y_train = load_labels(TRAIN_LABELS)

# # X_test = load_images(TEST_IMAGES)
# # y_test = load_labels(TEST_LABELS)

# # # перемешиваем обучающую выборку
# # train_indexes = np.arange(len(X_train))
# # np.random.shuffle(train_indexes)

# # X_train = X_train[train_indexes]
# # y_train = y_train[train_indexes]

# # # перемешиваем тестовую выборку
# # test_indexes = np.arange(len(X_test))
# # np.random.shuffle(test_indexes)

# # X_test = X_test[test_indexes]
# # y_test = y_test[test_indexes]


# # X_train = X_train[:30000]
# # y_train = y_train[:30000]

# # X_test = X_test[:5000]
# # y_test = y_test[:5000]

# # model = CNN(num_classes=26)

# # epochs = 3
# # learning_rate = 0.01

# # print("Начинаем обучение на буквах...")

# # for epoch in range(epochs):
# #     total_loss = 0

# #     indexes = np.arange(len(X_train))
# #     np.random.shuffle(indexes)

# #     for i in indexes:
# #         x = X_train[i].reshape(1, -1)
# #         target = np.array([y_train[i]])

# #         output = model.forward(x)

# #         loss = cross_entropy_loss(output, target)
# #         total_loss += loss

# #         grad = cross_entropy_grad(output, target)
# #         model.backward(grad, learning_rate)

# #     avg_loss = total_loss / len(X_train)

# #     correct = 0
# #     for i in range(len(X_test)):
# #         x = X_test[i].reshape(1, -1)
# #         pred = model.predict(x)[0]

# #         if pred == y_test[i]:
# #             correct += 1

# #     accuracy = correct / len(X_test)

# #     # print("Первые 20 правильных меток:", y_test[:20])

# #     # test_preds = []
# #     # for i in range(20):
# #     #     x = X_test[i].reshape(1, -1)
# #     #     pred = model.predict(x)[0]
# #     #     test_preds.append(pred)

# #     # print("Первые 20 предсказаний:", test_preds)

# #     print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}")

# # model.save("models/emnist_cnn_letters_model.npz")
# # print("Модель сохранена в models/emnist_cnn_letters_model.npz")


# import os
# import gzip
# import struct
# import numpy as np
# import cv2

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

#         images = np.transpose(images, (0, 2, 1))

#         return images.reshape(num, rows * cols) / 255.0


# def load_labels(path):
#     with gzip.open(path, "rb") as f:
#         magic, num = struct.unpack(">II", f.read(8))
#         labels = np.frombuffer(f.read(), dtype=np.uint8)
#         return labels - 1


# def augment_image(flat_image):
#     img = flat_image.reshape(28, 28)

#     # случайный поворот
#     angle = np.random.uniform(-15, 15)
#     matrix = cv2.getRotationMatrix2D((14, 14), angle, 1.0)
#     img = cv2.warpAffine(img, matrix, (28, 28), borderValue=0)

#     # случайный сдвиг
#     shift_x = np.random.randint(-3, 4)
#     shift_y = np.random.randint(-3, 4)
#     matrix = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
#     img = cv2.warpAffine(img, matrix, (28, 28), borderValue=0)

#     # byjulf утолщаем линии
#     if np.random.rand() < 0.5:
#         kernel = np.ones((2, 2), np.uint8)
#         img = cv2.dilate(img, kernel, iterations=1)

#     # иногда слегка размываем
#     if np.random.rand() < 0.3:
#         img = cv2.GaussianBlur(img, (3, 3), 0)

#     img = np.clip(img, 0, 1)

#     return img.reshape(784)


# def check_no_overlap(X1, X2, name1, name2):
#     set1 = set(row.tobytes() for row in (X1 * 255).astype(np.uint8))
#     set2 = set(row.tobytes() for row in (X2 * 255).astype(np.uint8))

#     overlap = len(set1.intersection(set2))

#     print(f"Пересечение {name1} и {name2}: {overlap}")

#     if overlap > 0:
#         raise ValueError(f"Ошибка: найдено пересечение между {name1} и {name2}")


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


# # делим train на train и validation
# validation_size = int(len(X_train) * 0.15)

# X_val = X_train[:validation_size]
# y_val = y_train[:validation_size]

# X_train = X_train[validation_size:]
# y_train = y_train[validation_size:]


# # перемешиваем тестовую выборку
# test_indexes = np.arange(len(X_test))
# np.random.shuffle(test_indexes)

# X_test = X_test[test_indexes]
# y_test = y_test[test_indexes]


# # ограничиваем объём данных, чтобы обучение на NumPy не шло слишком долго
# X_train = X_train[:30000]
# y_train = y_train[:30000]

# X_val = X_val[:5000]
# y_val = y_val[:5000]

# X_test = X_test[:5000]
# y_test = y_test[:5000]


# # проверяем, что используемые данные не пересекаются
# check_no_overlap(X_train, X_val, "train", "validation")
# check_no_overlap(X_train, X_test, "train", "test")
# check_no_overlap(X_val, X_test, "validation", "test")


# model = CNN(num_classes=26)

# epochs = 5
# learning_rate = 0.01

# print("Начинаем обучение CNN на буквах...")

# for epoch in range(epochs):
#     total_loss = 0

#     indexes = np.arange(len(X_train))
#     np.random.shuffle(indexes)

#     for i in indexes:
#         # аугментация применяется только к обучающей выборке
#         x_aug = augment_image(X_train[i])
#         x = x_aug.reshape(1, -1)

#         target = np.array([y_train[i]])

#         output = model.forward(x)

#         loss = cross_entropy_loss(output, target)
#         total_loss += loss

#         grad = cross_entropy_grad(output, target)
#         model.backward(grad, learning_rate)

#     avg_loss = total_loss / len(X_train)

#     # validation accuracy считаем без аугментации
#     correct = 0

#     for i in range(len(X_val)):
#         x = X_val[i].reshape(1, -1)
#         pred = model.predict(x)[0]

#         if pred == y_val[i]:
#             correct += 1

#     val_accuracy = correct / len(X_val)

#     print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}, Val Accuracy: {val_accuracy:.2f}")


# # финальная проверка на test
# correct = 0

# for i in range(len(X_test)):
#     x = X_test[i].reshape(1, -1)
#     pred = model.predict(x)[0]

#     if pred == y_test[i]:
#         correct += 1

# test_accuracy = correct / len(X_test)

# print(f"Final Test Accuracy: {test_accuracy:.2f}")

# model.save("models/emnist_cnn_letters_model.npz")
# print("Модель сохранена в models/emnist_cnn_letters_model.npz")




import os
import gzip
import struct
import csv
import time
import numpy as np
import cv2

from model import CNN
from loss import cross_entropy_loss, cross_entropy_grad


# ============================================================
# НАСТРОЙКИ ОБУЧЕНИЯ
# ============================================================

DATA_DIR = "data/emnist/gzip"

TRAIN_IMAGES = os.path.join(DATA_DIR, "emnist-letters-train-images-idx3-ubyte.gz")
TRAIN_LABELS = os.path.join(DATA_DIR, "emnist-letters-train-labels-idx1-ubyte.gz")
TEST_IMAGES = os.path.join(DATA_DIR, "emnist-letters-test-images-idx3-ubyte.gz")
TEST_LABELS = os.path.join(DATA_DIR, "emnist-letters-test-labels-idx1-ubyte.gz")

MODEL_DIR = "models"
LATEST_MODEL_PATH = os.path.join(MODEL_DIR, "emnist_cnn_letters_model.npz")
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "emnist_cnn_letters_best_model.npz")
LOG_PATH = os.path.join(MODEL_DIR, "training_log_letters.csv")

NUM_CLASSES = 26

# Полное обучение на train:
MAX_TRAIN = None

# Валидация и тест ограничены, чтобы проверка после эпохи не занимала слишком много времени.
MAX_VAL = 5000
MAX_TEST = 5000

EPOCHS = 5
LEARNING_RATE = 0.01

VALIDATION_RATIO = 0.15

USE_AUGMENTATION = True
CHECK_OVERLAP = True

RANDOM_SEED = 42

# Если обучение прервалось и нужно продолжить с последней сохранённой модели,
# поставить True.
RESUME_FROM_LATEST = False


# ============================================================
# ЗАГРУЗКА ДАННЫХ
# ============================================================

def load_images(path):
    with gzip.open(path, "rb") as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        data = np.frombuffer(f.read(), dtype=np.uint8)
        images = data.reshape(num, rows, cols)

        # ВАЖНО:
        # для EMNIST Letters корректная ориентация в нашем проекте —
        # только транспонирование, без flip.
        images = np.transpose(images, (0, 2, 1))

        return images.reshape(num, rows * cols) / 255.0


def load_labels(path):
    with gzip.open(path, "rb") as f:
        magic, num = struct.unpack(">II", f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)

        # EMNIST Letters хранит классы как 1–26.
        # Переводим в 0–25.
        return labels - 1


# ============================================================
# АУГМЕНТАЦИЯ
# ============================================================

def augment_image(flat_image):
    img = flat_image.reshape(28, 28)

    # Случайный небольшой поворот.
    angle = np.random.uniform(-12, 12)
    matrix = cv2.getRotationMatrix2D((14, 14), angle, 1.0)
    img = cv2.warpAffine(img, matrix, (28, 28), borderValue=0)

    # Случайный небольшой сдвиг.
    shift_x = np.random.randint(-2, 3)
    shift_y = np.random.randint(-2, 3)
    matrix = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    img = cv2.warpAffine(img, matrix, (28, 28), borderValue=0)

    # Иногда немного утолщаем линии.
    if np.random.rand() < 0.35:
        kernel = np.ones((2, 2), np.uint8)
        img = cv2.dilate(img, kernel, iterations=1)

    # Иногда слегка сглаживаем края.
    if np.random.rand() < 0.20:
        img = cv2.GaussianBlur(img, (3, 3), 0)

    img = np.clip(img, 0, 1)

    return img.reshape(784)


# ============================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================

def set_model_learning_rate(model, learning_rate):
    """
    DenseLayer получает learning_rate через backward,
    а ConvLayer хранит learning_rate внутри себя.
    Поэтому явно обновляем learning_rate у свёрточных слоёв.
    """
    model.conv1.learning_rate = learning_rate
    model.conv2.learning_rate = learning_rate


def check_no_overlap(X1, X2, name1, name2):
    set1 = set(row.tobytes() for row in (X1 * 255).astype(np.uint8))
    set2 = set(row.tobytes() for row in (X2 * 255).astype(np.uint8))

    overlap = len(set1.intersection(set2))

    print(f"Пересечение {name1} и {name2}: {overlap}")

    if overlap > 0:
        raise ValueError(f"Ошибка: найдено пересечение между {name1} и {name2}")


def evaluate(model, X, y):
    correct = 0

    for i in range(len(X)):
        x = X[i].reshape(1, -1)
        pred = model.predict(x)[0]

        if pred == y[i]:
            correct += 1

    return correct / len(X)


def save_log_header():
    if not os.path.exists(LOG_PATH):
        with open(LOG_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch",
                "loss",
                "val_accuracy",
                "epoch_time_seconds"
            ])


def append_log(epoch, loss, val_accuracy, epoch_time):
    with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            epoch,
            loss,
            val_accuracy,
            epoch_time
        ])


def validate_required_files():
    required_files = [
        TRAIN_IMAGES,
        TRAIN_LABELS,
        TEST_IMAGES,
        TEST_LABELS,
    ]

    missing = [path for path in required_files if not os.path.exists(path)]

    if missing:
        print("Не найдены файлы EMNIST:")
        for path in missing:
            print(" -", path)

        raise FileNotFoundError(
            "Проверь, что архив EMNIST распакован в папку data/emnist/gzip"
        )


# ============================================================
# ОСНОВНОЙ КОД
# ============================================================

def main():
    np.random.seed(RANDOM_SEED)
    os.makedirs(MODEL_DIR, exist_ok=True)

    validate_required_files()

    print("Загрузка EMNIST Letters...")

    X_train_full = load_images(TRAIN_IMAGES)
    y_train_full = load_labels(TRAIN_LABELS)

    X_test = load_images(TEST_IMAGES)
    y_test = load_labels(TEST_LABELS)

    print("Размер исходного train:", len(X_train_full))
    print("Размер исходного test:", len(X_test))

    # Перемешиваем train.
    train_indexes = np.arange(len(X_train_full))
    np.random.shuffle(train_indexes)

    X_train_full = X_train_full[train_indexes]
    y_train_full = y_train_full[train_indexes]

    # Делим train на train и validation.
    validation_size = int(len(X_train_full) * VALIDATION_RATIO)

    X_val = X_train_full[:validation_size]
    y_val = y_train_full[:validation_size]

    X_train = X_train_full[validation_size:]
    y_train = y_train_full[validation_size:]

    # Перемешиваем test отдельно.
    test_indexes = np.arange(len(X_test))
    np.random.shuffle(test_indexes)

    X_test = X_test[test_indexes]
    y_test = y_test[test_indexes]

    # Ограничения объёма, если нужны.
    if MAX_TRAIN is not None:
        X_train = X_train[:MAX_TRAIN]
        y_train = y_train[:MAX_TRAIN]

    if MAX_VAL is not None:
        X_val = X_val[:MAX_VAL]
        y_val = y_val[:MAX_VAL]

    if MAX_TEST is not None:
        X_test = X_test[:MAX_TEST]
        y_test = y_test[:MAX_TEST]

    print("Итоговый train:", len(X_train))
    print("Итоговый validation:", len(X_val))
    print("Итоговый test:", len(X_test))

    if CHECK_OVERLAP:
        print("Проверка пересечений...")
        check_no_overlap(X_train, X_val, "train", "validation")
        check_no_overlap(X_train, X_test, "train", "test")
        check_no_overlap(X_val, X_test, "validation", "test")

    model = CNN(num_classes=NUM_CLASSES)
    set_model_learning_rate(model, LEARNING_RATE)

    if RESUME_FROM_LATEST and os.path.exists(LATEST_MODEL_PATH):
        print("Загружаю существующую модель:", LATEST_MODEL_PATH)
        model.load(LATEST_MODEL_PATH)

    save_log_header()

    best_val_accuracy = 0.0

    print("Начинаем обучение CNN на EMNIST Letters...")
    print("Эпох:", EPOCHS)
    print("Learning rate:", LEARNING_RATE)
    print("Аугментация:", USE_AUGMENTATION)
    print("Последняя модель будет сохраняться в:", LATEST_MODEL_PATH)
    print("Лучшая модель будет сохраняться в:", BEST_MODEL_PATH)

    for epoch in range(EPOCHS):
        epoch_start = time.time()
        total_loss = 0.0

        indexes = np.arange(len(X_train))
        np.random.shuffle(indexes)

        for counter, i in enumerate(indexes, start=1):
            if USE_AUGMENTATION:
                x_train = augment_image(X_train[i])
            else:
                x_train = X_train[i]

            x = x_train.reshape(1, -1)
            target = np.array([y_train[i]])

            output = model.forward(x)

            loss = cross_entropy_loss(output, target)
            total_loss += loss

            grad = cross_entropy_grad(output, target)
            model.backward(grad, LEARNING_RATE)

            # Прогресс в консоли, чтобы было видно, что процесс не завис.
            if counter % 5000 == 0:
                print(f"  обработано {counter}/{len(X_train)} примеров")

        avg_loss = total_loss / len(X_train)
        val_accuracy = evaluate(model, X_val, y_val)
        epoch_time = time.time() - epoch_start

        print(
            f"Epoch {epoch + 1}/{EPOCHS}, "
            f"Loss: {avg_loss:.4f}, "
            f"Val Accuracy: {val_accuracy:.4f}, "
            f"Time: {epoch_time / 60:.1f} min"
        )

        # Сохраняем последнюю модель после каждой эпохи.
        model.save(LATEST_MODEL_PATH)
        print("Последняя модель сохранена:", LATEST_MODEL_PATH)

        # Сохраняем лучшую модель отдельно.
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            model.save(BEST_MODEL_PATH)
            print("Новая лучшая модель сохранена:", BEST_MODEL_PATH)

        append_log(epoch + 1, avg_loss, val_accuracy, epoch_time)

    print("Финальная проверка на test...")

    # Загружаем лучшую модель для финального теста.
    if os.path.exists(BEST_MODEL_PATH):
        model.load(BEST_MODEL_PATH)

    test_accuracy = evaluate(model, X_test, y_test)

    print(f"Final Test Accuracy: {test_accuracy:.4f}")

    # Сохраняем лучшую модель как основную для app.py.
    model.save(LATEST_MODEL_PATH)

    print("Финальная модель сохранена:", LATEST_MODEL_PATH)
    print("Лучшая модель:", BEST_MODEL_PATH)
    print("Лог обучения:", LOG_PATH)


if __name__ == "__main__":
    main()
