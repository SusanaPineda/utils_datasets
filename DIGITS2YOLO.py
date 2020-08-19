import os
import cv2
import numpy as np

URL_output = "../Datasets/complete/labels_YOLO/"
URL_DIGITS = "../Datasets/complete/labels_DIGITS/"
URL_IMGs = "../Datasets/complete/images/"
data = os.listdir(URL_DIGITS)

tags = np.array(['Peaton_verde', 'Peaton_rojo', 'Peaton_generico', 'Coche_verde', 'Coche_rojo', 'Coche_generico'])

for txt in data:
    img = cv2.imread(os.path.join(URL_IMGs, txt.split('.')[0] + ".png"))
    if img is None:
        img = cv2.imread(os.path.join(URL_IMGs, txt.split('.')[0] + ".jpg"))

    h = img.shape[0]
    w = img.shape[1]
    f = open(os.path.join(URL_DIGITS, txt))
    out = open(os.path.join(URL_output, txt), 'w')
    text = f.read()
    text = text.split()
    for i in range(int(len(text) / 15)):
        despl = i * 15
        tag = text[0 + despl]
        indx = np.where(tags == tag)
        x = (float(text[4 + despl]) + (float(text[6 + despl]) - float(text[4 + despl])) / 2) / w
        w_det = (float(text[6 + despl]) - float(text[4 + despl])) / w
        y = (float(text[5 + despl]) + (float(text[7 + despl]) - float(text[5 + despl])) / 2) / h
        h_det = (float(text[7 + despl]) - float(text[5 + despl])) / h
        out.write(str(indx[0][0]) + " " + str(x) + " " + str(y) + " " + str(w_det) + " " + str(h_det) + "\n")
    f.close()
    out.close()
