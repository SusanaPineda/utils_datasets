import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

images_URL = "../Datasets/data_5/TestyTrain/images/"
labels_URL = "../Datasets/data_5/TestyTrain/labels/"

test_images_URL = "../Datasets/data_5/test/images/"
test_labels_URL = "../Datasets/data_5/test/labels/"

train_images_URL = "../Datasets/data_5/train/images/"
train_labels_URL = "../Datasets/data_5/train/labels/"

#p_train = 0.8
cont = 20

data = os.listdir(labels_URL)

train, test = train_test_split(data, test_size=0.3, random_state=23)

for tr in train:
    img = cv2.imread(os.path.join(images_URL, tr.split('.')[0] + '.png'))
    if img is None:
        img = cv2.imread(os.path.join(images_URL, tr.split('.')[0] + '.jpg'))

    cv2.imwrite(os.path.join(train_images_URL, tr.split('.')[0]+'.png'), img)

    f = open(os.path.join(labels_URL, tr))
    out = open(os.path.join(train_labels_URL, tr), 'w')
    text = f.read()
    out.write(text)

    f.close()
    out.close()

for te in test:
    img = cv2.imread(os.path.join(images_URL, te.split('.')[0] + '.png'))

    cv2.imwrite(os.path.join(test_images_URL, te.split('.')[0]+'.png'), img)

    f = open(os.path.join(labels_URL, te))
    out = open(os.path.join(test_labels_URL, te), 'w')
    text = f.read()
    out.write(text)

    f.close()
    out.close()
