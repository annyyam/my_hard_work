import cv2
from image_processing import preprocess_image, extract_characters

image_path = "test.png" 

thresh = preprocess_image(image_path)

boxes = extract_characters(thresh)

img = cv2.imread(image_path)

for (x, y, w, h) in boxes:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

cv2.imshow("Detected Characters", img)
cv2.waitKey(0)
cv2.destroyAllWindows()