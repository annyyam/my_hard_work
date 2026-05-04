import matplotlib.pyplot as plt

from image_processing import preprocess_image, extract_characters, extract_and_prepare


image_path = "testKVMA.png"

thresh = preprocess_image(image_path)
boxes = extract_characters(thresh)
characters = extract_and_prepare(thresh, boxes)

print("Найдено символов:", len(characters))

for i, char in enumerate(characters):
    img = char.reshape(28, 28)

    plt.imshow(img, cmap="gray")
    plt.title(f"Символ {i + 1}")
    plt.show()