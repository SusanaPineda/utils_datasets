import os
import cv2
import numpy as np

"""
Metodos para modificar las etiquetas del dataset
"""


def update_info(url_input, url_output, tp, tg_c0, tg, d):
    data_digits = os.listdir(url_input)
    for txt_digits in data_digits:
        f = open(os.path.join(url_input, txt_digits))
        out = open(os.path.join(url_output, txt_digits), 'w')

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


def digits_1class(url_output_digits_labels, url_input_digits_labels, tag):
    """
    Modifica las etiquetas de un conjunto de datos en DIGITS para que contengan unicamente 1 clase
    :param url_output_digits_labels:
    :param url_input_digits_labels:
    :return:
    """

    update_info(url_input_digits_labels, url_output_digits_labels, "digits", tag, tag, 15)


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


def load_img(txt, url_imgs):
    img = cv2.imread(os.path.join(url_imgs, txt.split('.')[0] + ".png"))
    if img is None:
        img = cv2.imread(os.path.join(url_imgs, txt.split('.')[0] + ".jpg"))

    h = img.shape[0]
    w = img.shape[1]
    f = open(os.path.join(url_imgs, txt))
    text = f.read()
    text = text.split()
    f.close()
    return h, text, w


def digits_2_yolo(url_output, url_digits, url_imgs, tags):
    """
    Paso de etiquetas en digits a yolo
    :param url_output: directorio de salida de las etiquetas en yolo
    :param url_digits: dorectorio de las etiquetas en digits
    :param url_imgs: directorio de las imagenes
    :param tags: nombre de las clases
    :return:
    """
    data = os.listdir(url_digits)
    for txt in data:
        out = open(os.path.join(url_output, txt), 'w+')
        h, text, w = load_img(txt, url_imgs)

        for i in range(int(len(text) / 15)):
            despl = i * 15
            tag = text[0 + despl]
            indx = np.where(tags == tag)
            x = (float(text[4 + despl]) + (float(text[6 + despl]) - float(text[4 + despl])) / 2) / w
            w_det = (float(text[6 + despl]) - float(text[4 + despl])) / w
            y = (float(text[5 + despl]) + (float(text[7 + despl]) - float(text[5 + despl])) / 2) / h
            h_det = (float(text[7 + despl]) - float(text[5 + despl])) / h
            out.write(str(indx[0][0]) + " " + str(x) + " " + str(y) + " " + str(w_det) + " " + str(h_det) + "\n")

        out.close()


def yolo_2_digits_makesense(url_output, url_yolo, url_imgs, tags):
    """
    Paso de las etiquetas obtenidas con makesense en YOLO a DIGITS
    :param url_output: url del directorio de salida de las etiquetas
    :param url_yolo: url del directorio de las etiquetas en yolo
    :param url_imgs: url de las imagenes
    :param tags: clases
    :return:
    """

    data = os.listdir(url_yolo)
    for txt in data:
        out = open(os.path.join(url_output, txt), 'w')
        h, text, w = load_img(txt, url_imgs)
        for i in range(int(len(text) / 5)):
            despl = i * 5
            tag = tags[int(text[0 + despl])]
            left = (float(text[1 + despl]) * w) - (float(text[3 + despl]) * w / 2)
            top = (float(text[2 + despl]) * h) - (float(text[4 + despl]) * h / 2)
            right = (float(text[1 + despl]) * w) + (float(text[3 + despl]) * w / 2)
            bottom = (float(text[2 + despl]) * h) + (float(text[4 + despl]) * h / 2)
            out.write(str(tag) + " 0 " + "0 " + "0 " + str(left) + " " + str(top) + " " + str(right) + " " + str(bottom)
                      + " 0 " + "0 " + "0 " + "0 " + "0 " + "0 " + "0" + "\n")
        out.close()


def count_class(url_labels, url_output, tags, d):
    """
    Conteo del numero de detecciones de cada clase presentes en uno o varios directorios de etiquetas.
    :param url_labels: directorios a analizar
    :param url_output: directorio en el que almacenar el resultado
    :param tags: etiquetas de las clases
    :param d: desplazamiento a aplicar 15 para digits, 5 para yolo
    :return:
    """
    count = np.zeros(len(tags))

    out = open(os.path.join(url_output, "clases.txt"), 'w')

    for dir in url_labels:
        data = os.listdir(dir)
        for txt in data:
            f = open(os.path.join(dir, txt))
            text = f.read()
            text = text.split()
            for i in range(int(len(text) / d)):
                despl = i * d
                tag = text[0 + despl]
                indx = np.where(tags == tag)
                count[indx[0]] = count[indx[0]] + 1
            f.close()
        print(dir)
    print(count)
    out.write(str(count))
    out.close()

    print("finish")