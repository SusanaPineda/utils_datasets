import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

images_URL = "/home/susi/Documents/Datasets/Barcelona/images/"
labels_URL = "/home/susi/Documents/Datasets/Barcelona/labels_DIGITS/"

test_images_URL = "../Datasets/data_8/test/images/"
test_labels_URL = "../Datasets/data_8/test/labels_DIGITS/"

train_images_URL = "../Datasets/data_8/train/images/"
train_labels_URL = "../Datasets/data_8/train/labels_DIGITS/"

val_images_URL = "../Datasets/data_8/val/images/"
val_labels_URL = "../Datasets/data_8/val/labels_DIGITS/"

cont = 20

data = os.listdir(images_URL)
tt = []

for d in data:
    if (d.split('_')[0] == "BG") and (d.split('_')[1] == "IMG"):
        img = cv2.imread(os.path.join(images_URL, d.split('.')[0] + '.png'))
        if img is None:
            img = cv2.imread(os.path.join(images_URL, d.split('.')[0] + '.jpg'))

        cv2.imwrite(os.path.join(val_images_URL, d.split('.')[0] + '.png'), img)

        if os.path.exists(os.path.join(labels_URL, d.split('.')[0] + '.txt')):
            f = open(os.path.join(labels_URL, d.split('.')[0] + '.txt'))
        else:
            f = open(os.path.join(labels_URL, d.split('.')[0] + '.txt'), 'w+')
        out = open(os.path.join(val_labels_URL, d.split('.')[0] + '.txt'), 'w')
        text = f.read()
        out.write(text)

        f.close()
        out.close()
    else:
        tt.append(d)

print("Validación terminada")
train, test = train_test_split(tt, test_size=0.1, random_state=23)

for tr in train:
    """if (((tr.split("_")[0] == "CAPO") and (tr.split("_")[1] != "IMG")) or (tr.split("_")[0] == "SVO") or (tr.split("_")[0] == "ITALIA")) and (cont == 0):
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

        cont = cont - 1
    else:"""
    img = cv2.imread(os.path.join(images_URL, tr.split('.')[0] + '.png'))
    if img is None:
        img = cv2.imread(os.path.join(images_URL, tr.split('.')[0] + '.jpg'))

    cv2.imwrite(os.path.join(train_images_URL, tr.split('.')[0] + '.png'), img)

    if os.path.exists(os.path.join(labels_URL, tr.split('.')[0] + '.txt')):
        f = open(os.path.join(labels_URL, tr.split('.')[0] + '.txt'))
    else:
        f = open(os.path.join(labels_URL, tr.split('.')[0] + '.txt'), 'w+')
    out = open(os.path.join(train_labels_URL, tr.split('.')[0] + '.txt'), 'w')
    text = f.read()
    out.write(text)

    f.close()
    out.close()

print("Train terminada")
for te in test:
    """if (((te.split("_")[0] == "CAPO") and (te.split("_")[1] != "IMG")) or (te.split("_")[0] == "SVO") or (te.split("_")[0] == "ITALIA")) and (cont2 == 0):
        img = cv2.imread(os.path.join(images_URL, te.split('.')[0] + '.png'))
        if img is None:
            img = cv2.imread(os.path.join(images_URL, te.split('.')[0] + '.jpg'))

        cv2.imwrite(os.path.join(test_images_URL, te.split('.')[0]+'.png'), img)

        f = open(os.path.join(labels_URL, te))
        out = open(os.path.join(test_labels_URL, te), 'w')
        text = f.read()
        out.write(text)

        f.close()
        out.close()

        cont2 = cont2-1
    else:"""
    img = cv2.imread(os.path.join(images_URL, te.split('.')[0] + '.png'))
    if img is None:
        img = cv2.imread(os.path.join(images_URL, te.split('.')[0] + '.jpg'))

    cv2.imwrite(os.path.join(test_images_URL, te.split('.')[0] + '.png'), img)

    if os.path.exists(os.path.join(labels_URL, te.split('.')[0] + '.txt')):
        f = open(os.path.join(labels_URL, te.split('.')[0] + '.txt'))
    else:
        f = open(os.path.join(labels_URL, te.split('.')[0] + '.txt'), 'w+')
    out = open(os.path.join(test_labels_URL, te.split('.')[0] + '.txt'), 'w')
    text = f.read()
    out.write(text)

    f.close()
    out.close()
