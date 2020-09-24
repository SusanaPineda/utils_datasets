import numpy as np
import os
import cv2
import matplotlib.pyplot as plt

URL_ground_truth = "/home/susi/Documents/Datasets/data_8/val/labels_YOLO/"
URL_results = "/home/susi/Documents/Pruebas_BG/D5_5e-05_2_3_0995_600/"
URL_images = "/home/susi/Documents/Datasets/data_8/val/images/"

labels = np.array(['Peaton_verde', 'Peaton_rojo', 'Peaton_generico', 'Coche_verde', 'Coche_rojo', 'Coche_generico'])
#labels = np.array(['Peaton', 'Coche'])

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
            cn = (float(text[2 + despl]) * h, float(text[1 + despl]) * w)
        else:
            tag = int(text[0 + despl])
            left = float(text[1 + despl])
            top = float(text[2 + despl])
            width = float(text[3 + despl]) - left
            height = float(text[4 + despl]) - top
            cn = (top + height / 2, left + width / 2)

        cl.append(tag)
        center.append(cn)

    return cl, center


def paint_detections(cl_r, pr, cl_gt, pgt, im):
    i = 0
    for p in pr:
        cv2.circle(im, (int(p[1]), int(p[0])), 3, (255, 0, 0), 2)
        cv2.putText(im, str(cl_r[i]), (int(p[1]) + 5, int(p[0]) + 5), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 0, 0), 2)
        i = i + 1
    i = 0
    for g in pgt:
        cv2.circle(im, (int(g[1]), int(g[0])), 3, (0, 0, 255), 2)
        cv2.putText(im, str(cl_gt[i]), (int(g[1]) + 5, int(g[0]) + 5), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 255), 2)
        i = i + 1

    #im = cv2.resize(im, (int(im.shape[1]*0.2), int(im.shape[0]*0.2)))
    cv2.imshow("compare", im)
    cv2.waitKey(0)


def compare(cl_r, ps_r, cl_gt, ps_gt, d):
    i = 0
    cont = len(cl_gt)
    sums = np.zeros((labels.size+1, labels.size+1))

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
        if dist < d:
            sums[cl_res][cl_gth] = sums[cl_res][cl_gth] + 1
            cl_r = np.delete(cl_r, dis_min)
            cl_gt = np.delete(cl_gt, i)
            ps_r = np.delete(ps_r, dis_min, 0)
            ps_gt = np.delete(ps_gt, i, 0)
        else:
            i = i + 1

    while len(cl_gt) > 0:
        sums[labels.size][cl_gt[0]] = sums[labels.size][cl_gt[0]] + 1
        cl_gt = np.delete(cl_gt, 0)

    while len(cl_r) > 0:
        sums[cl_r[0]][labels.size] = sums[cl_r[0]][labels.size] + 1
        cl_r = np.delete(cl_r, 0)

    return sums


def get_numbers(m, labels):
    divs = m.sum(axis=0)
    for i in range(len(labels)):
        tp = m[i][i] / divs[i]
        fp = m[i][labels.size] / divs[i]
        fn = m[labels.size][i] / divs[i]
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
total_matrix = np.zeros((labels.size+1, labels.size+1))
vis = False

for d in data:

    results = open(os.path.join(URL_results, d))
    gt = open(os.path.join(URL_ground_truth, d))

    img = cv2.imread(os.path.join(URL_images, d.split('.')[0] + ".png"))

    distance = img.shape[0]/5

    class_results, positions_results = get_info(results, img, "results")
    class_gt, positions_gt = get_info(gt, img, "gt")

    if vis:
        paint_detections(class_results, positions_results, class_gt, positions_gt, img.copy())

    matrix = compare(class_results.copy(), positions_results.copy(), class_gt.copy(), positions_gt.copy(), distance)

    total_matrix = total_matrix + matrix

plt.matshow(total_matrix)
plt.colorbar()
plt.xlabel("groundTruth")
plt.ylabel("results")

plt.show()

get_numbers(total_matrix, labels)
