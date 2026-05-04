import cv2
import numpy as np


def preprocess_image(image_path):
    img = cv2.imread(image_path)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    thresh = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        21,
        8
    )

    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.dilate(thresh, kernel, iterations=2)

    return thresh

# def preprocess_image(image_path):
#     # загрузка изображения
#     img = cv2.imread(image_path)

#     # перевод в серый
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#     # бинаризация (чёрно-белое)
#     _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

#     return thresh


def extract_characters(thresh):
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        # ФИЛЬТР МУСОРА
        if w < 10 or h < 10:
            continue

        boxes.append((x, y, w, h))

    # сортируем слева направо
    boxes = sorted(boxes, key=lambda b: b[0])

    return boxes


# def extract_and_prepare(thresh, boxes):
#     characters = []

#     for (x, y, w, h) in boxes:
#         char = thresh[y:y+h, x:x+w]

#         # делаем квадрат
#         size = max(w, h)
#         square = np.zeros((size, size))

#         # центрируем символ в квадрате
#         x_offset = (size - w) // 2
#         y_offset = (size - h) // 2

#         square[y_offset:y_offset+h, x_offset:x_offset+w] = char

#         #уменьшаем до 20x20
#         char_resized = cv2.resize(square, (20, 20))

#         #создаём 28x28
#         canvas = np.zeros((28, 28))

#         # вставляем в центр
#         canvas[4:24, 4:24] = char_resized

#         # нормализация
#         canvas = canvas / 255.0

#         characters.append(canvas.reshape(1, 784))

#     return characters

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

        # центрируем по центру масс
        coords = np.column_stack(np.where(char_resized > 0))

        if len(coords) > 0:
            y_mean, x_mean = coords.mean(axis=0)

            shift_x = int(10 - x_mean)
            shift_y = int(10 - y_mean)

            M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
            char_resized = cv2.warpAffine(char_resized, M, (20, 20))

        # создаём 28x28
        canvas = np.zeros((28, 28))
        canvas[4:24, 4:24] = char_resized

        # нормализация
        canvas = canvas / 255.0

        characters.append(canvas.reshape(1, 784))

    return characters