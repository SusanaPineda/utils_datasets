import os
import cv2

input_URL_images = "/home/susi/Documents/Datasets/validacion_semaforos/images"
input_URL_labels = "/home/susi/Documents/Datasets/validacion_semaforos/labels_results"


def get_labels(file, img):
    text = file.read()
    text = text.split()
    for i in range(int(len(text) / 5)):
        despl = i * 5
        tag = text[0 + despl]
        left = float(text[1 + despl])
        top = float(text[2 + despl])
        right = float(text[3 + despl])
        bottom = float(text[4 + despl])

        img = cv2.rectangle(img, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), 3)
    cv2.imshow("augment", img)
    cv2.waitKey(0)


data = os.listdir(input_URL_labels)

for d in data:
    file = open(os.path.join(input_URL_labels, d))

    if len(d.split('.')[0].split('_')) == 3:
        img = cv2.imread(os.path.join(input_URL_images, d.split('.')[0] + ".jpg"))
    else:
        img = cv2.imread(os.path.join(input_URL_images, d.split('.')[0] + ".png"))

    get_labels(file, img)





