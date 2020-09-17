import os
from PIL import Image
import cv2
import numpy as np

input_URL_images = "/home/susi/Documents/Datasets/val2/images/"
input_URL_labels = "/home/susi/Documents/Datasets/val2/labels/"


output_URL_images = "/home/susi/Documents/Datasets/val2_resize640/images/"
output_URL_labels = "/home/susi/Documents/Datasets/val2_resize640/labels/"


def update_labels(f, res, o):
    text = f.read()
    text = text.split()

    for i in range(int(len(text) / 15)):
        despl = i * 15
        tag = text[0 + despl]
        left = float(text[4 + despl]) * res[1]
        top = float(text[5 + despl]) * res[0]
        right = float(text[6 + despl]) * res[1]
        bottom = float(text[7 + despl]) * res[0]
        o.write(str(tag) + " 0 " + "0 " + "0 " + str(left) + " " + str(top) + " " + str(right) + " " + str(bottom)
                + " 0 " + "0 " + "0 " + "0 " + "0 " + "0 " + "0" + "\n")



def apply_transform(img, res):

    im = img.resize((640, 640))
    #im = img.resize((int(img.width * res), int(img.height * res)))

    return im


data = os.listdir(input_URL_labels)

for d in data:
    file = open(os.path.join(input_URL_labels, d))

    out_dir_labels = os.path.join(output_URL_labels, d)
    out_dir_images = os.path.join(output_URL_images, d.split('.')[0] + '.png')

    out = open(out_dir_labels, 'w')

    img = Image.open(os.path.join(input_URL_images, d.split('.')[0] + '.png'))

    resize = [1, 1]
    if img.width > 1000:
        resize = [0.2, 0.2]

    h0 = img.height
    w0 = img.width

    img1 = apply_transform(img, resize)
    cv2.imwrite(out_dir_images, cv2.cvtColor(np.array(img1), cv2.COLOR_BGR2RGB))

    h1 = img1.height
    w1 = img1.width

    if w1 != w0:
        resize[0] = 1/(h0/h1)
        resize[1] = 1/(w0/w1)

    update_labels(file, resize, out)
    file.close()
    out.close()
