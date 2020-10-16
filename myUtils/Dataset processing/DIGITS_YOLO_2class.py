import os
import numpy as np


def update_info(url_input, url_output, tp, tg_c0, tg, d):
    data_DIGITS = os.listdir(url_input)
    for txt_DIGITS in data_DIGITS:
        f = open(os.path.join(url_input, txt_DIGITS))
        out = open(os.path.join(url_output, txt_DIGITS), 'w')

        text = f.read()
        text = text.split()
        for i in range(int(len(text) / d)):
            despl = i * d
            left = float(text[4 + despl])
            top = float(text[5 + despl])
            right = float(text[6 + despl])
            bottom = float(text[7 + despl])
            tag = text[0 + despl]
            if np.any(tg_c0 == tag):
                if tp == "digits":
                    out.write(str(tg[0]) + " 0 " + "0 " + "0 " + str(left) + " " + str(top) + " " + str(right) + " " +
                              str(bottom) + " 0 " + "0 " + "0 " + "0 " + "0 " + "0 " + "0" + "\n")
                else:
                    out.write("0 " + str(left) + " " + str(top) + " " + str(right) + " " + str(bottom) + "\n")
            else:
                if tp == "digits":
                    out.write(str(tg[1]) + " 0 " + "0 " + "0 " + str(left) + " " + str(top) + " " + str(right) + " " +
                              str(bottom) + " 0 " + "0 " + "0 " + "0 " + "0 " + "0 " + "0" + "\n")
                else:
                    out.write("1 " + str(left) + " " + str(top) + " " + str(right) + " " + str(bottom) + "\n")
        f.close()
        out.close()


def digits_yolo_2class(url_output_digits_labels, url_output_yolo_labels, url_input_digits_labels, url_input_yolo_labels,
                       yolo, digits, tags_c0, tags):
    """
    Modifica las etiquetas de un conjunto de datos en DIGITS o YOLO para que contengan 2 clases
    :param url_output_digits_labels: url en la que se almacenaran las etiquetas en digits
    :param url_output_yolo_labels: url en la que se almacenaran las etiquetas en yolo
    :param url_input_digits_labels: url en la que se encuentran las etiquetas en DIGITS
    :param url_input_yolo_labels: url en la que se encuentran las etiquetas en YOLO
    :param yolo: indica si se van a tranformar etiquetas en formato YOLO
    :param digits: indica si se van a transformar etiquetas en formato DIGITS
    :param tags_c0: numpy array con las etiquetas que pasaran a ser de la clase 0
    :param tags: etiquetas DIGITS finales
    :return:
    """

    if digits:
        update_info(url_input_digits_labels, url_output_digits_labels, "digits", tags_c0, tags, 15)
    if yolo:
        update_info(url_input_yolo_labels, url_output_yolo_labels, "yolo", tags_c0, tags, 5)
