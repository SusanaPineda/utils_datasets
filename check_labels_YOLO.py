import os
import cv2

input_URL_images = "/home/susi/Documents/Datasets/Alex_val/images/"
input_URL_labels = "/home/susi/Documents/Datasets/Alex_val/labels_YOLO_2class"


def get_labels(file, img):
    text = file.read()
    text = text.split()
    h = img.shape[0]
    w = img.shape[1]
    for i in range(int(len(text) / 5)):
        despl = i * 5
        tag = text[0 + despl]
        """left = float(text[1 + despl])
        top = float(text[2 + despl])
        right = float(text[3 + despl])
        bottom = float(text[4 + despl])"""
        left = (float(text[1 + despl]) * w) - ((float(text[3 + despl]) * w) / 2)
        top = (float(text[2 + despl]) * h) - ((float(text[4 + despl]) * h) / 2)
        right = (float(text[1 + despl]) * w) + ((float(text[3 + despl]) * w) / 2)
        bottom = (float(text[2 + despl]) * h) + ((float(text[4 + despl]) * h) / 2)

        img = cv2.rectangle(img, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), 3)

    #img = cv2.resize(img, (int(img.shape[1] * 0.2), int(img.shape[0] * 0.2)))
    cv2.imshow("augment", img)
    cv2.waitKey(0)


data = os.listdir(input_URL_labels)

for d in data:
    file = open(os.path.join(input_URL_labels, d))
    print(file)

    img = cv2.imread(os.path.join(input_URL_images, d.split('.')[0] + ".png"))
    if img is None:
        img = cv2.imread(os.path.join(input_URL_images, d.split('.')[0] + ".jpg"))

    get_labels(file, img)





