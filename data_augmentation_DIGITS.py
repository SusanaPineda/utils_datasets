import os
from PIL import Image, ImageFilter, ImageDraw
import random
import cv2
import numpy as np

input_URL_images = "/home/susi/Documents/Datasets/data_2/train/images"
input_URL_labels = "/home/susi/Documents/Datasets/data_2/train/labels"


output_URL_images = "/home/susi/Documents/Datasets/augmentation/images"
output_URL_labels = "/home/susi/Documents/Datasets/augmentation/labels"


def update_labels(f, res, img, o):
    text = f.read()
    text = text.split()
    if res < 0.25:
        res = 1

    for i in range(int(len(text) / 15)):
        despl = i * 15
        tag = text[0 + despl]
        left = float(text[4 + despl]) * res
        top = float(text[5 + despl]) * res
        right = float(text[6 + despl]) * res
        bottom = float(text[7 + despl]) * res
        o.write(str(tag) + " 0 " + "0 " + "0 " + str(left) + " " + str(top) + " " + str(right) + " " + str(bottom)
                + " 0 " + "0 " + "0 " + "0 " + "0 " + "0 " + "0" + "\n")
        img = cv2.rectangle(img, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), 3)

    return img


def get_random_transform():
    b = random.randint(2, 4)  # 4 == False
    res = random.uniform(0, 1.5)  # <0.25 = false

    return b, res


def apply_transform(img, b, res):
    if b != 4:
        img = img.filter(ImageFilter.GaussianBlur(b))
    if res >= 0.25:
        img = img.resize((int(img.width * res), int(img.height * res)))
    return img


data = os.listdir(input_URL_labels)
for d in data:
    file = open(os.path.join(input_URL_labels, d))
    if d.split('_')[0] == 'ITALIA':
        out_dir_labels = os.path.join(output_URL_labels, d.split('_')[0] + '_AU_' + d.split('_')[1])
        out_dir_images = os.path.join(output_URL_images, d.split('_')[0] + '_AU_' + d.split('_')[1].split('.')[0] + '.png')
    else:
        out_dir_labels = os.path.join(output_URL_labels, d.split('_')[0] + '_AU_' + d.split('_')[1] + '_' + d.split('_')[2])
        out_dir_images = os.path.join(output_URL_images, d.split('_')[0] + '_AU_' + d.split('_')[1] + '_' + d.split('_')[2].split('.')[0] + '.png')

    out = open(out_dir_labels, 'w')

    img = Image.open(os.path.join(input_URL_images, d.split('.')[0] + '.png'))

    blur, resize = get_random_transform()
    img1 = apply_transform(img, blur, resize)
    cv2.imwrite(out_dir_images, cv2.cvtColor(np.array(img1), cv2.COLOR_BGR2RGB))

    image = update_labels(file, resize, np.array(img1), out)
    file.close()
    out.close()

    '''cv2.imshow("augment", cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break'''
