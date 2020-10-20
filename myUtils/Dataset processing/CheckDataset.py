import os
import cv2


def visualize_data(img):
    cv2.imshow("img", img)
    cv2.waitKey(0)


def save_data(img, output_url, name):
    cv2.imwrite(os.path.join(output_url, name.split('.')[0] + ".png"), img)


def get_yolo_coordinates(despl, h, text, w):
    """
    Obtención de las coordenadas de las detecciones en yolo
    :param despl: desplazamiento para la lectura de las etiquetas
    :param h: altura de la imagen
    :param text: archivo a leer
    :param w: ancho de la imagen
    :return:
    """
    tag = int(text[0 + despl])
    left = (float(text[1 + despl]) * w) - ((float(text[3 + despl]) * w) / 2)
    top = (float(text[2 + despl]) * h) - ((float(text[4 + despl]) * h) / 2)
    right = (float(text[1 + despl]) * w) + ((float(text[3 + despl]) * w) / 2)
    bottom = (float(text[2 + despl]) * h) + ((float(text[4 + despl]) * h) / 2)
    return bottom, left, right, top, tag


def get_digits_coordinates(despl, text):
    """
    Obtención de las coordenadas de las detecciones en digits
    :param despl: desplazamiento para la lectura de las etiquetas
    :param text: archivo a leer
    :return:
    """
    left = float(text[4 + despl])
    top = float(text[5 + despl])
    right = float(text[6 + despl])
    bottom = float(text[7 + despl])
    return bottom, left, right, top


def get_det_coordinates(despl, text):
    """
    Obtención de las coordenadas de las detecciones del groundtruth
    :param despl: desplazamiento para la lectura de las etiquetas
    :param text: archivo a leer
    :return:
    """
    tag = int(text[0 + despl])
    left = float(text[1 + despl])
    top = float(text[2 + despl])
    right = float(text[3 + despl])
    bottom = float(text[4 + despl])
    return bottom, left, right, top, tag


def get_labels(file, im, d, color, tipo):
    """
    Recorrer etiquetas y marcar las detecciones de cada una de las imágenes
    :param file: fichero en el que se encuentran las etiquetas
    :param im: imagen
    :param d: desplazamiento en las etiquetas
    :return:
    """
    text = file.read()
    text = text.split()
    h = im.shape[0]
    w = im.shape[1]
    for i in range(int(len(text) / d)):
        despl = i * d
        if tipo == "yolo":
            bottom, left, right, top, tag = get_yolo_coordinates(despl, h, text, w)
        elif tipo == "digits":
            tag = 0
            bottom, left, right, top = get_digits_coordinates(despl, text)
        elif tipo == "det":
            bottom, left, right, top, tag = get_det_coordinates(despl, text)

        im = cv2.rectangle(im, (int(left), int(top)), (int(right), int(bottom)), color[tag], 3)
    return im


def check_labels(input_url_images, input_url_labels, output_url, ds, tipo, save, visualize, color):
    """
    Comprobar las etiquetas en YOLO o DIGITS
    :param input_url_images: url de las imágenes
    :param input_url_labels: url de las etiquetas
    :param ds: desplazamiento. 5 = YOLO, 15 = DIGITS
    :return:
    """
    data = os.listdir(input_url_labels)

    for d in data:
        file = open(os.path.join(input_url_labels, d))

        img = cv2.imread(os.path.join(input_url_images, d.split('.')[0] + '.png'))
        if img is None:
            img = cv2.imread(os.path.join(input_url_images, d.split('.')[0] + ".jpg"))

        img = get_labels(file, img, ds, color, tipo)

        if visualize:
            visualize_data(img)
        if save:
            save_data(img, output_url, d)


def compare_results(input_url_images, input_url_results_labels, input_url_gt_labels, output_url, colors_gt,
                    colors_result, save, visualize):
    """
    Comparar los resultados obtenidos
    :param input_url_images:
    :param input_url_results_labels:
    :param input_url_gt_labels:
    :param output_url:
    :param colors_gt:
    :param colors_result:
    :param save:
    :param visualize:
    :return:
    """
    data = os.listdir(input_url_gt_labels)

    for d in data:
        file = open(os.path.join(input_url_results_labels, d))
        file_gt = open(os.path.join(input_url_gt_labels, d))

        img = cv2.imread(os.path.join(input_url_images, d.split('.')[0] + ".png"))
        if img is None:
            img = cv2.imread(os.path.join(input_url_images, d.split('.')[0] + ".jpg"))

        img = get_labels(file_gt, img, 5, colors_gt, "yolo")
        img = get_labels(file, img, 5, colors_result, "det")

        if visualize:
            visualize_data(img)

        if save:
            save_data(img, output_url, d)
