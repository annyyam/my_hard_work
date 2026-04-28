from model import NeuralNetwork
from image_processing import preprocess_image, extract_characters, extract_and_prepare


image_path = "test.png"

model = NeuralNetwork()
#model.load("models/digits_model.npz")
model.load("models/mnist_model.npz")

thresh = preprocess_image(image_path)
boxes = extract_characters(thresh)
characters = extract_and_prepare(thresh, boxes)

result = ""

for char in characters:
    prediction = model.predict(char)[0]
    result += str(prediction)

print("Распознанный текст:", result)