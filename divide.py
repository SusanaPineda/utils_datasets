import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

images_URL = "/home/susi/Documents/Datasets/data_8/train/images/"
labels_URL = "/home/susi/Documents/Datasets/data_8/train/labels_2class_YOLO/"

test_images_URL = "/home/susi/Documents/Datasets/data_8/val2/images/"
test_labels_URL = "/home/susi/Documents/Datasets/data_8/val2/labels/"

"""train_images_URL = "/media/susi/B48C43F88C43B420/Datasets/data_8/train/images/"
train_labels_URL = "/media/susi/B48C43F88C43B420/Datasets/data_8/train/labels/"""""

#p_train = 0.8
cont = 20

data = os.listdir(images_URL)

train, test = train_test_split(data, test_size=0.05, random_state=23)

"""for tr in train:
    img = cv2.imread(os.path.join(images_URL, tr.split('.')[0] + '.png'))
    if img is None:
        img = cv2.imread(os.path.join(images_URL, tr.split('.')[0] + '.jpg'))

    cv2.imwrite(os.path.join(train_images_URL, tr.split('.')[0]+'.png'), img)

    if os.path.exists(os.path.join(labels_URL, tr.split('.')[0] + '.txt')):
        f = open(os.path.join(labels_URL, tr.split('.')[0] + '.txt'))
    else:
        f = open(os.path.join(labels_URL, tr.split('.')[0] + '.txt'), 'w+')

    out = open(os.path.join(train_labels_URL, tr.split('.')[0] + '.txt'), 'w')
    text = f.read()
    out.write(text)

    f.close()
    out.close()"""

for te in test:
    img = cv2.imread(os.path.join(images_URL, te.split('.')[0] + '.png'))
    if img is None:
        img = cv2.imread(os.path.join(images_URL, te.split('.')[0] + '.jpg'))

    cv2.imwrite(os.path.join(test_images_URL, te.split('.')[0]+'.png'), img)

    if os.path.exists(os.path.join(labels_URL, te.split('.')[0] + '.txt')):
        f = open(os.path.join(labels_URL, te.split('.')[0] + '.txt'))
    else:
        f = open(os.path.join(labels_URL, te.split('.')[0] + '.txt'), 'w+')

    out = open(os.path.join(test_labels_URL, te.split('.')[0] + '.txt'), 'w')
    text = f.read()
    out.write(text)

    f.close()
    out.close()
