import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

import cv2
import numpy as np

from model import CNN
from image_processing import preprocess_image, extract_characters, extract_and_prepare


model = CNN(num_classes=26)
model.load("models/emnist_cnn_letters_model.npz")
# model = NeuralNetwork()
# model.load("models/mnist_model.npz")

def scale_to_canvas(img, scale_x=1.0, scale_y=1.0):
    h, w = img.shape

    new_w = max(1, int(w * scale_x))
    new_h = max(1, int(h * scale_y))

    resized = cv2.resize(img, (new_w, new_h))

    canvas = np.zeros((28, 28))

    if new_w > 28:
        start_x = (new_w - 28) // 2
        resized = resized[:, start_x:start_x + 28]
        new_w = 28

    if new_h > 28:
        start_y = (new_h - 28) // 2
        resized = resized[start_y:start_y + 28, :]
        new_h = 28

    x_offset = (28 - new_w) // 2
    y_offset = (28 - new_h) // 2

    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

    return canvas


def predict_with_augmentation(char):
    img = char.reshape(28, 28)

    variants = []

    # исходный вариант
    variants.append(img)

    # небольшие сдвиги
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        matrix = np.float32([[1, 0, dx], [0, 1, dy]])
        shifted = cv2.warpAffine(img, matrix, (28, 28), borderValue=0)
        variants.append(shifted)

    # небольшие повороты
    for angle in [-5, 5]:
        matrix = cv2.getRotationMatrix2D((14, 14), angle, 1.0)
        rotated = cv2.warpAffine(img, matrix, (28, 28), borderValue=0)
        variants.append(rotated)

    # слегка шире — помогает буквам M, W, A, но не слишком сильно
    variants.append(scale_to_canvas(img, scale_x=1.12, scale_y=1.0))

    # слегка толще
    kernel = np.ones((2, 2), np.uint8)
    thicker = cv2.dilate((img * 255).astype(np.uint8), kernel, iterations=1) / 255.0
    variants.append(thicker)

    all_probs = []

    for variant in variants:
        x = variant.reshape(1, 784)
        probs = model.forward(x)[0]
        all_probs.append(probs)

    # берём только среднее, без max_probs
    final_probs = np.mean(all_probs, axis=0)

    # нормализуем, чтобы проценты были корректными
    final_probs = final_probs / np.sum(final_probs)

    return final_probs

def load_image():
    file_path = filedialog.askopenfilename()

    if not file_path:
        return

    # показать картинку
    img = Image.open(file_path)
    img = img.resize((200, 200))
    img_tk = ImageTk.PhotoImage(img)

    panel.config(image=img_tk)
    panel.image = img_tk

    # распознавание
    thresh = preprocess_image(file_path)
    boxes = extract_characters(thresh)
    chars = extract_and_prepare(thresh, boxes)

    result = ""
    details = []

    for char in chars:
        # probs = model.forward(char)[0]
        probs = predict_with_augmentation(char)

        top_indexes = probs.argsort()[-3:][::-1]

        top_letters = []
        for idx in top_indexes:
            letter = chr(idx + ord("A"))
            percent = probs[idx] * 100
            top_letters.append(f"{letter} ({percent:.1f}%)")

        pred = top_indexes[0]
        result += chr(pred + ord("A"))

        details.append(", ".join(top_letters))

    result_label.config(text="Результат: " + result)

    print("Top-3 по символам:")
    for i, d in enumerate(details):
        print(f"Символ {i + 1}: {d}")


root = tk.Tk()
root.title("Распознавание рукописного текста")

btn = tk.Button(root, text="Загрузить изображение", command=load_image)
btn.pack()

panel = tk.Label(root)
panel.pack()

result_label = tk.Label(root, text="Результат: ", font=("Arial", 16))
result_label.pack()

root.mainloop()