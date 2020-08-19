import numpy as np
import os
import cv2
import matplotlib.pyplot as plt

URL_ground_truth = "../Datasets/val_TFM/2clases/labels_YOLO/"
URL_results = "../Datasets/val_TFM/2class_2D/"
URL_images = "../Datasets/val_TFM/images/"

"""URL_ground_truth = "../Datasets/val_5/labels_YOLO/"
URL_results = "../Datasets/validacion_semaforos/Detect_semaphore_dataset7_00006_2_3_0995_340_newclass/"
URL_images = "../Datasets/val_5/images/" """

def get_info(file, im, str):
    cl = []
    center = []
    h = im.shape[0]
    w = im.shape[1]
    text = file.read()
    text = text.split()
    for i in range(int(len(text) / 5)):
        despl = i * 5

        if str == "gt":
            tag = int(text[0 + despl])
            cn = (float(text[1 + despl]) * w, float(text[2 + despl]) * h)
            '''left = (float(text[1+despl])*w)-(float(text[3+despl]*w)/2)
            top = (float(text[2+despl])*h)-(float(text[4+despl]*h)/2)
            right = (float(text[1+despl])*w)+(float(text[3+despl]*w)/2)
            bottom = (float(text[2+despl])*h)+(float(text[4+despl]*h)/2)'''
        else:
            tag = int(text[0 + despl])
            top = float(text[1 + despl])
            left = float(text[2 + despl])
            width = float(text[3 + despl]) - top
            height = float(text[4 + despl]) - left
            cn = (top + height / 2, left + width / 2)

        cl.append(tag)
        center.append(cn)

    return cl, center


def paint_detections(cl_r, pr, cl_gt, pgt, im):
    i = 0
    for p in pr:
        cv2.circle(im, (int(p[0]), int(p[1])), 3, (255, 0, 0), 2)
        cv2.putText(im, str(cl_r[i]), (int(p[0]) + 5, int(p[1]) + 5), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 0, 0), 2)
        i = i + 1
    i = 0
    for g in pgt:
        cv2.circle(im, (int(g[0]), int(g[1])), 3, (0, 0, 255), 2)
        cv2.putText(im, str(cl_gt[i]), (int(g[0]) + 5, int(g[1]) + 5), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 255), 2)
        i = i + 1

    cv2.imshow("compare", im)
    cv2.waitKey(0)


def compare(cl_r, ps_r, cl_gt, ps_gt):
    i = 0
    cont = len(cl_gt)
    sums = np.zeros((8, 8))

    while (len(cl_r) > 0) & (len(cl_gt) > 0) & (cont > 0):
        cont = cont - 1
        np_pos = np.asarray(ps_gt[i])
        # diff = np.absolute(np.subtract(ps_r, np_pos))
        diff = [np.linalg.norm(np.asarray(x) - np_pos) for x in list(ps_r)]
        dis_min = np.argmin(diff)
        # dis_min = int(dis_min/2)

        cl_gth = cl_gt[i]
        cl_res = cl_r[dis_min]
        dist = diff[dis_min]
        if dist < 90:
            sums[cl_res][cl_gth] = sums[cl_res][cl_gth] + 1
            cl_r = np.delete(cl_r, dis_min)
            cl_gt = np.delete(cl_gt, i)
            ps_r = np.delete(ps_r, dis_min, 0)
            ps_gt = np.delete(ps_gt, i, 0)
        else:
            i = i + 1

    while len(cl_gt) > 0:
        sums[7][cl_gt[0]] = sums[7][cl_gt[0]] + 1
        cl_gt = np.delete(cl_gt, 0)

    while len(cl_r) > 0:
        sums[cl_r[0]][6] = sums[cl_r[0]][6] + 1
        cl_r = np.delete(cl_r, 0)

    return sums


def get_numbers(m, labels):
    divs = m.sum(axis=0)
    for i in range(len(labels)):
        tp = m[i][i] / divs[i]
        fp = m[i][6] / divs[i]
        fn = m[7][i] / divs[i]
        ta = tp / (tp + fp + fn)
        te = (fp + fn) / (tp + fp + fn)
        p = tp / (tp + fp)
        print("\n--------------------")
        print("\nMétricas clase " + str(labels[i]))
        print("\n   True Positives: " + str(tp))
        print("\n   False Positives: " + str(fp))
        print("\n   False Negatives: " + str(fn))
        print("\n       Tasa de acierto: " + str(ta))
        print("\n       Tasa de error: " + str(te))
        print("\n       Tasa de precision: " + str(p))



data = os.listdir(URL_ground_truth)
total_matrix = np.zeros((8, 8))
vis = False

for d in data:

    results = open(os.path.join(URL_results, d))
    gt = open(os.path.join(URL_ground_truth, d))

    '''if len(d.split('.')[0].split('_')) == 3:
        img = cv2.imread(os.path.join(URL_images, d.split('.')[0] + ".jpg"))
    else:'''
    img = cv2.imread(os.path.join(URL_images, d.split('.')[0] + ".png"))

    class_results, positions_results = get_info(results, img, "results")
    class_gt, positions_gt = get_info(gt, img, "gt")

    if vis:
        paint_detections(class_results, positions_results, class_gt, positions_gt, img.copy())

    matrix = compare(class_results.copy(), positions_results.copy(), class_gt.copy(), positions_gt.copy())

    total_matrix = total_matrix + matrix

plt.matshow(total_matrix)
plt.colorbar()
plt.xlabel("groundTruth")
plt.ylabel("results")
# plt.title("Detect_semaphore_dataset5_00005_2_3_0995_600")
plt.show()

labels = np.array(['Peaton', 'Coche'])
#labels = np.array(['Peaton_verde', 'Peaton_rojo', 'Peaton_generico', 'Coche_verde', 'Coche_rojo', 'Coche_generico'])
get_numbers(total_matrix, labels)
