import os
import cv2

input_URL_images = "/home/susi/Documents/Datasets/val1_resize640/images/"
input_URL_labels = "/home/susi/Documents/Datasets/val1_resize640/labels/"


def get_labels(file, img):
    text = file.read()
    text = text.split()
    for i in range(int(len(text) / 15)):
        despl = i * 15
        tag = text[0 + despl]
        left = float(text[4 + despl])
        top = float(text[5 + despl])
        right = float(text[6 + despl])
        bottom = float(text[7 + despl])

        img = cv2.rectangle(img, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), 3)
    cv2.imshow("augment", img)
    cv2.waitKey(0)


data = os.listdir(input_URL_labels)

for d in data:
    file = open(os.path.join(input_URL_labels, d))

    img = cv2.imread(os.path.join(input_URL_images, d.split('.')[0] + '.png'))

    get_labels(file, img)





