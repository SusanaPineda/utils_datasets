import cv2
import os
import argparse

URL_carpetas = ["../Datasets/11-53-32/", "../Datasets/11-53-49/", "../Datasets/11-53-55/",
                "../Datasets/11-54-01/", "../Datasets/11-54-06/"]

URL_carpetas_SVOs = ["../Datasets/HD1080_SN10028700_16-28-49", "../Datasets/HD1080_SN10028700_16-31-51",
                     "../Datasets/HD1080_SN10028700_16-33-06", "../Datasets/HD1080_SN10028700_16-36-06"]

URL_output = "../Datasets/validacion_semaforos/images_2/"

for i in range(len(URL_carpetas)):
    data = os.listdir(URL_carpetas[i])
    cont = 30
    for img in data:
        if cont == 0:
            frame = cv2.imread(os.path.join(URL_carpetas[i], img))
            cv2.imwrite(os.path.join(URL_output, img), frame)
            cont = 30
        cont = cont - 1

for j in range(len(URL_carpetas_SVOs)):
    data = os.listdir(URL_carpetas_SVOs[j])
    cont = 50
    for img in data:
        if cont == 0:
            frame = cv2.imread(os.path.join(URL_carpetas_SVOs[j], img))
            cv2.imwrite(os.path.join(URL_output, img), frame)
            cont = 50
        cont = cont - 1
