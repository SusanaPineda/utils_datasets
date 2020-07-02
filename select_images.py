import cv2
import os
import argparse

URL_carpetas = ["./11-53-32/", "./11-53-49/", "./11-53-55/", "./11-54-01/",
                "./11-54-06/", "./PTLR_dataset/Training_set/Italia/"]

URL_carpetas_SVOs = ["./HD1080_SN10028700_16-28-49/", "./HD1080_SN10028700_16-31-51/",
                     "./HD1080_SN10028700_16-33-06/", "./HD1080_SN10028700_16-36-06/"]

URL_output = "./dataset_CAPO/"

for i in range(len(URL_carpetas)):
    data = os.listdir(URL_carpetas[i])
    cont = 3
    for img in data:
        if cont == 0:
            frame = cv2.imread(os.path.join(URL_carpetas[i], img))
            cv2.imwrite(os.path.join(URL_output, img), frame)
            cont = 3
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
