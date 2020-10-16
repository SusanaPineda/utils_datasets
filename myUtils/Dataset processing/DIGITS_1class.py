import os


def digits_1class(url_output_digits_labels, url_input_digits_labels, tag):
    """
    Modifica las etiquetas de un conjunto de datos en DIGITS para que contengan unicamente 1 clase
    :param url_output_digits_labels:
    :param url_input_digits_labels:
    :return:
    """

    data_DIGITS = os.listdir(url_input_digits_labels)

    for txt_DIGITS in data_DIGITS:
        f = open(os.path.join(url_input_digits_labels, txt_DIGITS))
        out = open(os.path.join(url_output_digits_labels, txt_DIGITS), 'w')

        text = f.read()
        text = text.split()
        for i in range(int(len(text) / 15)):
            despl = i * 15
            left = float(text[4 + despl])
            top = float(text[5 + despl])
            right = float(text[6 + despl])
            bottom = float(text[7 + despl])

            out.write(str(tag) + " 0 " + "0 " + "0 " + str(left) + " " + str(top) + " " + str(right) + " " + str(bottom)
                      + " 0 " + "0 " + "0 " + "0 " + "0 " + "0 " + "0" + "\n")

        f.close()
        out.close()
