import os
import cv2
import numpy as np

URL_output_DIGITS_labels = "../Datasets/data_2/train/labels_2/"
#URL_output_YOLO_labels = ""

URL_input_DIGITS_labels = "../Datasets/data_2/train/labels/"
#URL_input_YOLO_labels = ""

data_DIGITS = os.listdir(URL_input_DIGITS_labels)
#data_YOLO = os.listdir(URL_input_YOLO_labels)

tags = np.array(['Peaton_verde', 'Peaton_rojo', 'Peaton_generico', 'Coche_verde', 'Coche_rojo', 'Coche_generico'])

for txt_DIGITS in data_DIGITS:
    f = open(os.path.join(URL_input_DIGITS_labels, txt_DIGITS))
    out = open(os.path.join(URL_output_DIGITS_labels, txt_DIGITS), 'w')

    text = f.read()
    text = text.split()
    for i in range(int(len(text) / 15)):
        despl = i * 15
        left = float(text[4 + despl])
        top = float(text[5 + despl])
        right = float(text[6 + despl])
        bottom = float(text[7 + despl])
        tag = text[0 + despl]
        indx = np.where(tags == tag)
        if (indx[0][0] == 0) or (indx[0][0] == 1) or (indx[0][0] == 2):
            out.write("Peaton" + " 0 " + "0 " + "0 " + str(left) + " " + str(top) + " " + str(right) + " " + str(bottom)
                      + " 0 " + "0 " + "0 " + "0 " + "0 " + "0 " + "0" + "\n")
        else:
            out.write("Coche" + " 0 " + "0 " + "0 " + str(left) + " " + str(top) + " " + str(right) + " " + str(bottom)
                      + " 0 " + "0 " + "0 " + "0 " + "0 " + "0 " + "0" + "\n")
    f.close()
    out.close()

"""for txt_YOLO in data_YOLO:
    f = open(os.path.join(URL_input_YOLO_labels, txt_YOLO))
    out = open(os.path.join(URL_output_YOLO_labels, txt_YOLO), 'w')

    text_YOLO = f.read()
    text_YOLO = text_YOLO.split()
    for i in range(int(len(text_YOLO) / 5)):
        d = i * 5
        l = float(text_YOLO[1 + d])
        t = float(text_YOLO[2 + d])
        r = float(text_YOLO[3 + d])
        b = float(text_YOLO[4 + d])
        tag = int(text_YOLO[0 + d])
        if (tag == 0) or (tag == 1) or (tag == 2):
            out.write("0 " + str(l) + " " + str(t) + " " + str(r) + " " + str(b) + "\n")
        else:
            out.write("1 " + str(l) + " " + str(t) + " " + str(r) + " " + str(b) + "\n")
    f.close()
    out.close()"""