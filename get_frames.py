import cv2
import os
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-v","--video", required = True, help = "path del video")
ap.add_argument("-s","--salida", required = True, help = "path de la carpeta de salida")
ap.add_argument("-e","--etiqueta", required = True, help = "etiqueta")

args = vars(ap.parse_args())

inputURL = args['video']
outputURL = args['salida']
etiqueta = args['etiqueta']

cap = cv2.VideoCapture(inputURL)
cont = 0
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        cv2.imshow("frame", frame)
        name = os.path.join(outputURL, etiqueta)
        cv2.imwrite(name+str(cont)+".png", frame)
        cont = cont+1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
