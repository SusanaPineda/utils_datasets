import os

URL_output_DIGITS_labels = "/home/susi/Documents/Datasets/data_8/val/labels_1class/"

URL_input_DIGITS_labels = "/home/susi/Documents/Datasets/data_8/val/labels_2class/"

data_DIGITS = os.listdir(URL_input_DIGITS_labels)

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

        out.write("Semaforo" + " 0 " + "0 " + "0 " + str(left) + " " + str(top) + " " + str(right) + " " + str(bottom)
                  + " 0 " + "0 " + "0 " + "0 " + "0 " + "0 " + "0" + "\n")

    f.close()
    out.close()

