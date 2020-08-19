import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

images_URL = "/media/susi/B48C43F88C43B420/Datasets/complete/images/"
labels_URL = "/media/susi/B48C43F88C43B420/Datasets/complete/labels_DIGITS/"

test_images_URL = "../Datasets/data_7/test/images/"
test_labels_URL = "../Datasets/data_7/test/labels/"

train_images_URL = "../Datasets/data_7/train/images/"
train_labels_URL = "../Datasets/data_7/train/labels/"

val_images_URL = "../Datasets/data_7/val/images/"
val_labels_URL = "../Datasets/data_7/val/labels/"

cont = 20

data = os.listdir(labels_URL)
tt = []

for d in data:
    if (d.split('_')[0] == "BG") and (d.split('_')[1] == "IMG"):
        img = cv2.imread(os.path.join(images_URL, d.split('.')[0] + '.png'))
        if img is None:
            img = cv2.imread(os.path.join(images_URL, d.split('.')[0] + '.jpg'))

        cv2.imwrite(os.path.join(val_images_URL, d.split('.')[0] + '.png'), img)

        f = open(os.path.join(labels_URL, d))
        out = open(os.path.join(val_labels_URL, d), 'w')
        text = f.read()
        out.write(text)

        f.close()
        out.close()
    else:
        tt.append(d)

print("Valaidación terminada")
train, test = train_test_split(tt, test_size=0.1, random_state=23)

for tr in train:
    if (((tr.split("_")[0] == "CAPO") and (tr.split("_")[1] != "IMG")) or (tr.split("_")[0] == "SVO") or (tr.split("_")[0] == "ITALIA")) and (cont == 0):
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
    else:
        img = cv2.imread(os.path.join(images_URL, tr.split('.')[0] + '.png'))
        if img is None:
            img = cv2.imread(os.path.join(images_URL, tr.split('.')[0] + '.jpg'))

        cv2.imwrite(os.path.join(train_images_URL, tr.split('.')[0] + '.png'), img)

        f = open(os.path.join(labels_URL, tr))
        out = open(os.path.join(train_labels_URL, tr), 'w')
        text = f.read()
        out.write(text)

        f.close()
        out.close()

cont2 = 20

print("Train terminada")
for te in test:
    if (((te.split("_")[0] == "CAPO") and (te.split("_")[1] != "IMG")) or (te.split("_")[0] == "SVO") or (te.split("_")[0] == "ITALIA")) and (cont2 == 0):
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
    else:
        img = cv2.imread(os.path.join(images_URL, te.split('.')[0] + '.png'))
        if img is None:
            img = cv2.imread(os.path.join(images_URL, te.split('.')[0] + '.jpg'))

        cv2.imwrite(os.path.join(test_images_URL, te.split('.')[0] + '.png'), img)

        f = open(os.path.join(labels_URL, te))
        out = open(os.path.join(test_labels_URL, te), 'w')
        text = f.read()
        out.write(text)

        f.close()
        out.close()