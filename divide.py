import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

images_URL = "../Datasets/dataset_CAPO/"
labels_URL = "../Datasets/DIGITS/"

test_images_URL = "../Datasets/test/images/"
test_labels_URL = "../Datasets/test/labels/"

train_images_URL = "../Datasets/train/images/"
train_labels_URL = "../Datasets/train/labels/"

p_train = 0.8

data = os.listdir(labels_URL)

train, test = train_test_split(data, test_size=0.30, random_state=23)

for tr in train:
    if tr.split('_')[0] == 'ITALIA':
        img = cv2.imread(os.path.join(images_URL, tr.split('.')[0] + '.jpg'))
    else:
        img = cv2.imread(os.path.join(images_URL, tr.split('.')[0] + '.png'))

    cv2.imwrite(os.path.join(train_images_URL, tr.split('.')[0]+'.png'), img)

    f = open(os.path.join(labels_URL, tr))
    out = open(os.path.join(train_labels_URL, tr), 'w')
    text = f.read()
    out.write(text)

    f.close()
    out.close()

for te in test:
    if te.split('_')[0] == 'ITALIA':
        img = cv2.imread(os.path.join(images_URL, te.split('.')[0] + '.jpg'))
    else:
        img = cv2.imread(os.path.join(images_URL, te.split('.')[0] + '.png'))

    cv2.imwrite(os.path.join(test_images_URL, te.split('.')[0]+'.png'), img)

    f = open(os.path.join(labels_URL, te))
    out = open(os.path.join(test_labels_URL, te), 'w')
    text = f.read()
    out.write(text)

    f.close()
    out.close()
