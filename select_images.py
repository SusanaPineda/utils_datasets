import cv2
import os
import argparse

URL_carpetas = ["../Datasets/validacion_semaforos/frames/"]

URL_carpetas_SVOs = []

URL_output = "../Datasets/validacion_semaforos/images/"

for i in range(len(URL_carpetas)):
    data = os.listdir(URL_carpetas[i])
    cont = 10
    for img in data:
        if cont == 0:
            frame = cv2.imread(os.path.join(URL_carpetas[i], img))
            cv2.imwrite(os.path.join(URL_output, img), frame)
            cont = 10
        cont = cont - 1

for j in range(len(URL_carpetas_SVOs)):
    data = os.listdir(URL_carpetas_SVOs[j])
    cont = 4
    for img in data:
        if cont == 0:
            frame = cv2.imread(os.path.join(URL_carpetas_SVOs[j], img))
            cv2.imwrite(os.path.join(URL_output, img), frame)
            cont = 4
        cont = cont - 1
