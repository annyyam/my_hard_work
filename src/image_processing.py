import cv2
import numpy as np

def preprocess_image(image_path):
    img = cv2.imread(image_path)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # сглаживание
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # автоматический порог вместо фиксированного
    _, thresh = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    kernel = np.ones((2, 2), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)

    return thresh


# def preprocess_image(image_path):
#     img = cv2.imread(image_path)

#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#     gray = cv2.GaussianBlur(gray, (3, 3), 0)

#     thresh = cv2.adaptiveThreshold(
#         gray,
#         255,
#         cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#         cv2.THRESH_BINARY_INV,
#         21,
#         8
#     )

#     kernel = np.ones((3, 3), np.uint8)
#     thresh = cv2.dilate(thresh, kernel, iterations=2)

#     return thresh

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

    raw_boxes = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h

        # совсем мелкий мусор убираем
        if area < 20:
            continue

        raw_boxes.append((x, y, w, h))

    if not raw_boxes:
        return []

    # ориентируемся на самый высокий объект как на высоту обычной буквы
    max_h = max(h for x, y, w, h in raw_boxes)
    max_area = max(w * h for x, y, w, h in raw_boxes)

    big_boxes = []
    small_boxes = []

    for x, y, w, h in raw_boxes:
        area = w * h

        # маленькие элементы: точки над i/j
        if h < max_h * 0.45 and area < max_area * 0.30:
            small_boxes.append((x, y, w, h))
        else:
            big_boxes.append((x, y, w, h))

    # присоединяем маленькие элементы к ближайшей основной букве
    for sx, sy, sw, sh in small_boxes:
        s_center_x = sx + sw / 2
        s_center_y = sy + sh / 2

        best_index = None
        best_score = 10**9

        for i, (x, y, w, h) in enumerate(big_boxes):
            b_center_x = x + w / 2

            # точка должна быть примерно над буквой или в верхней её части
            if s_center_y > y + h * 0.65:
                continue

            # точка должна быть по горизонтали рядом с буквой
            if not (x - 25 <= s_center_x <= x + w + 25):
                continue

            horizontal_distance = abs(b_center_x - s_center_x)
            vertical_distance = abs(y - (sy + sh))

            score = horizontal_distance + vertical_distance * 0.5

            if score < best_score:
                best_score = score
                best_index = i

        if best_index is not None:
            x, y, w, h = big_boxes[best_index]

            new_x = min(x, sx)
            new_y = min(y, sy)
            new_right = max(x + w, sx + sw)
            new_bottom = max(y + h, sy + sh)

            big_boxes[best_index] = (
                new_x,
                new_y,
                new_right - new_x,
                new_bottom - new_y
            )

        # если маленький элемент не удалось присоединить,
        # считаем его шумом и не добавляем отдельно

    big_boxes = sorted(big_boxes, key=lambda b: b[0])

    return big_boxes

# def extract_characters(thresh):
#     contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     boxes = []

#     for cnt in contours:
#         x, y, w, h = cv2.boundingRect(cnt)

#         # ФИЛЬТР МУСОРА
#         if w < 10 or h < 10:
#             continue

#         boxes.append((x, y, w, h))

#     # сортируем слева направо
#     boxes = sorted(boxes, key=lambda b: b[0])

#     return boxes


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

        scale = 20.0 / max(w, h)
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))

        char_resized = cv2.resize(char, (new_w, new_h))

        # немного утолщаем линии, чтобы символы были похожи на EMNIST
        kernel = np.ones((2, 2), np.uint8)
        char_resized = cv2.dilate(char_resized, kernel, iterations=1)
        
        # слегка сглаживаем края, как у обучающих изображений
        char_resized = cv2.GaussianBlur(char_resized, (3, 3), 0)

        canvas = np.zeros((28, 28))

        x_offset = (28 - new_w) // 2
        y_offset = (28 - new_h) // 2

        canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = char_resized

        coords = np.column_stack(np.where(canvas > 0))

        if len(coords) > 0:
            y_mean, x_mean = coords.mean(axis=0)

            shift_x = int(14 - x_mean)
            shift_y = int(14 - y_mean)

            M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
            canvas = cv2.warpAffine(canvas, M, (28, 28), borderValue=0)

        canvas = canvas / 255.0

        characters.append(canvas.reshape(1, 784))

    return characters

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

#         # уменьшаем до 20x20
#         char_resized = cv2.resize(square, (20, 20))

#         # центрируем по центру масс
#         coords = np.column_stack(np.where(char_resized > 0))

#         if len(coords) > 0:
#             y_mean, x_mean = coords.mean(axis=0)

#             shift_x = int(10 - x_mean)
#             shift_y = int(10 - y_mean)

#             M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
#             char_resized = cv2.warpAffine(char_resized, M, (20, 20))


#         canvas = np.zeros((28, 28))
#         canvas[4:24, 4:24] = char_resized
#         canvas = canvas / 255.0

#         # # создаём 28x28
#         # canvas = np.zeros((28, 28))
#         # canvas[4:24, 4:24] = char_resized

#         # # нормализация
#         # canvas = canvas / 255.0

#         characters.append(canvas.reshape(1, 784))

#     return characters