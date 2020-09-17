import os
import cv2

URL_output = "/home/susi/Documents/Datasets/val2/labels/"
URL_YOLO = "/home/susi/Documents/Datasets/val2/labels_YOLO/"
URL_IMGs = "/home/susi/Documents/Datasets/val2/images/"
data = os.listdir(URL_YOLO)

tags = ['Peaton_verde', 'Peaton_rojo', 'Peaton_generico', 'Coche_verde', 'Coche_rojo', 'Coche_generico']

for txt in data:
    """if txt.split('_')[0] == 'ITALIA':
        img = cv2.imread(os.path.join(URL_IMGs, txt.split('.')[0] + '.jpg'))
    else:
        img = cv2.imread(os.path.join(URL_IMGs, txt.split('.')[0]+'.png'))
    if len(txt.split('.')[0].split('_')) == 3:
        img = cv2.imread(os.path.join(URL_IMGs, txt.split('.')[0] + ".jpg"))
    else:"""
    img = cv2.imread(os.path.join(URL_IMGs, txt.split('.')[0] + ".png"))

    h = img.shape[0]
    w = img.shape[1]
    f = open(os.path.join(URL_YOLO, txt))
    out = open(os.path.join(URL_output, txt), 'w')
    text = f.read()
    text = text.split()
    for i in range(int(len(text) / 5)):
        despl = i * 5
        tag = tags[int(text[0 + despl])]
        left = (float(text[1 + despl]) * w) - (float(text[3 + despl]) * w / 2)
        top = (float(text[2 + despl]) * h) - (float(text[4 + despl]) * h / 2)
        right = (float(text[1 + despl]) * w) + (float(text[3 + despl]) * w / 2)
        bottom = (float(text[2 + despl]) * h) + (float(text[4 + despl]) * h / 2)
        out.write(str(tag) + " 0 " + "0 " + "0 " + str(left) + " " + str(top) + " " + str(right) + " " + str(bottom)
                  + " 0 " + "0 " + "0 " + "0 " + "0 " + "0 " + "0" + "\n")
    f.close()
    out.close()
