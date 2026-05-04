import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

from model import CNN
from image_processing import preprocess_image, extract_characters, extract_and_prepare


model = CNN(num_classes=26)
model.load("models/emnist_cnn_letters_model.npz")
# model = NeuralNetwork()
# model.load("models/mnist_model.npz")


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
        probs = model.forward(char)[0]

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