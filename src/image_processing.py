import cv2
import numpy as np


def preprocess_image(image_path):
    # загрузка изображения
    img = cv2.imread(image_path)

    # перевод в серый
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # бинаризация (чёрно-белое)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

    return thresh


def extract_characters(thresh):
    # ищем контуры
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    boxes = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        # фильтр мусора
        if w > 5 and h > 5:
            boxes.append((x, y, w, h))

    # сортировка слева направо
    boxes = sorted(boxes, key=lambda b: b[0])

    return boxes


def extract_and_prepare(thresh, boxes):
    characters = []

    for (x, y, w, h) in boxes:
        char = thresh[y:y+h, x:x+w]

        # делаем квадрат
        size = max(w, h)
        square = np.zeros((size, size))

        # центрируем символ в квадрате
        x_offset = (size - w) // 2
        y_offset = (size - h) // 2

        square[y_offset:y_offset+h, x_offset:x_offset+w] = char

        # уменьшаем до 20x20
        char_resized = cv2.resize(square, (20, 20))

        # создаём 28x28
        canvas = np.zeros((28, 28))

        # вставляем в центр
        canvas[4:24, 4:24] = char_resized

        # нормализация
        canvas = canvas / 255.0

        characters.append(canvas.reshape(1, 784))

    return characters