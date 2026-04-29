import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

from model import NeuralNetwork
from image_processing import preprocess_image, extract_characters, extract_and_prepare


model = NeuralNetwork(num_classes=26)
model.load("models/emnist_letters_model.npz")
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

    for char in chars:
        pred = model.predict(char)[0]
        result += chr(pred + ord("A"))
        #result += str(pred)

    result_label.config(text="Результат: " + result)


root = tk.Tk()
root.title("Распознавание рукописного текста")

btn = tk.Button(root, text="Загрузить изображение", command=load_image)
btn.pack()

panel = tk.Label(root)
panel.pack()

result_label = tk.Label(root, text="Результат: ", font=("Arial", 16))
result_label.pack()

root.mainloop()