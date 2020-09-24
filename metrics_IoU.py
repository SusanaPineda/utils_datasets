import numpy as np
import os
import cv2
import matplotlib.pyplot as plt

URL_ground_truth = "/home/susi/Documents/Datasets/data_8/val/labels_DIGITS/"
URL_results = "/home/susi/Documents/Pruebas_BG/D5_5e-05_2_3_0995_600/"
URL_images = "/home/susi/Documents/Datasets/data_8/val/images/"

labels = np.array(['Peaton_verde', 'Peaton_rojo', 'Peaton_generico', 'Coche_verde', 'Coche_rojo', 'Coche_generico'])
colors_gt = [(8, 186, 255), (7, 163, 250), (4, 93, 232), (2, 47, 220), (0, 0, 208), (8, 2, 157)]
colors_results = [(244, 232, 173), (228, 202, 72), (216, 180, 0), (182, 119, 0), (138, 62, 2), (94, 4, 3)]
threshold_iou = 0.3


def get_intersection_area(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    return (xB - xA + 1) * (yB - yA + 1)


def get_union_areas(boxA, boxB, interArea):
    area_A = get_area(boxA)
    area_B = get_area(boxB)

    return float(area_A + area_B - interArea)


def get_area(box):
    return (box[2] - box[0] + 1) * (box[3] - box[1] + 1)


def get_iou(boxA, boxB):
    intersection = get_intersection_area(boxA, boxB)

    if intersection > 0:
        union = get_union_areas(boxA, boxB, intersection)
        return intersection / union
    else:
        return -1


def get_bbx(file, s):
    cl = []
    bbx = []
    text = file.read()
    text = text.split()

    if s == "gt":
        for i in range(int(len(text) / 15)):
            despl = i * 15
            tag = text[0 + despl]
            indx = np.where(labels == tag)
            left = float(text[4 + despl])
            top = float(text[5 + despl])
            right = float(text[6 + despl])
            bottom = float(text[7 + despl])

            cl.append(indx[0][0])
            bbx.append((int(left), int(top), int(right), int(bottom)))
    else:
        for i in range(int(len(text) / 5)):
            despl = i * 5
            indx = int(text[0 + despl])
            left = float(text[1 + despl])
            top = float(text[2 + despl])
            right = float(text[3 + despl])
            bottom = float(text[4 + despl])

            cl.append(indx)
            bbx.append((int(left), int(top), int(right), int(bottom)))

    return cl, bbx


def paint_detections(cl_r, pr, cl_gt, pgt, im):
    i = 0
    for p in pr:
        im = cv2.rectangle(im, (p[0], p[1]), (p[2], p[3]), colors_results[cl_r[i]], 5)
        cv2.putText(im, str(cl_r[i]), (p[0], p[1]), cv2.FONT_HERSHEY_COMPLEX, 0.6, colors_results[cl_r[i]], 2)
        i = i + 1
    i = 0
    for g in pgt:
        im = cv2.rectangle(im, (g[0], g[1]), (g[2], g[3]), colors_gt[int(cl_gt[i])], 5)
        cv2.putText(im, str(cl_gt[i]), (g[0], g[1]), cv2.FONT_HERSHEY_COMPLEX, 0.6, colors_gt[int(cl_gt[i])], 2)
        i = i + 1

    im = cv2.resize(im, (int(im.shape[1]*0.2), int(im.shape[0]*0.2)))
    cv2.imshow("compare", im)
    cv2.waitKey(0)


def compare(cl_r, ps_r, cl_gt, ps_gt):
    i = 0
    cont = len(cl_gt)
    sums = np.zeros((labels.size+1, labels.size+1))

    while ((len(cl_r) > 0) & (len(cl_gt) > 0) & (cont > 0)) & (i < len(cl_r)):
        aux = []
        cont = cont - 1
        bbx = np.asarray(ps_r[i])
        for p in ps_gt:
            aux.append(get_iou(bbx, p))
        idx_iou_max = np.argmax(aux)
        iou_max = aux[idx_iou_max]

        cl_gth = cl_gt[idx_iou_max]
        cl_res = cl_r[i]
        if iou_max > threshold_iou:
            sums[cl_res][cl_gth] = sums[cl_res][cl_gth] + 1
            cl_r = np.delete(cl_r, i)
            cl_gt = np.delete(cl_gt, idx_iou_max)
            ps_r = np.delete(ps_r, i, 0)
            ps_gt = np.delete(ps_gt, idx_iou_max, 0)
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
        r = tp /(tp + fn)
        print("\n--------------------")
        print("\nMétricas clase " + str(labels[i]))
        print("\n   True Positives: " + str(tp))
        print("\n   False Positives: " + str(fp))
        print("\n   False Negatives: " + str(fn))
        print("\n       Tasa de acierto: " + str(ta))
        print("\n       Tasa de error: " + str(te))
        print("\n       precision: " + str(p))
        print("\n       recall: " + str(r))


data = os.listdir(URL_ground_truth)
total_matrix = np.zeros((labels.size + 1, labels.size + 1))
vis = False

for d in data:

    results = open(os.path.join(URL_results, d))
    gt = open(os.path.join(URL_ground_truth, d))

    img = cv2.imread(os.path.join(URL_images, d.split('.')[0] + ".png"))

    class_results, positions_results = get_bbx(results, "results")
    class_gt, positions_gt = get_bbx(gt, "gt")

    if vis:
        paint_detections(class_results, positions_results, class_gt, positions_gt, img.copy())

    matrix = compare(class_results.copy(), positions_results.copy(), class_gt.copy(), positions_gt.copy())

    total_matrix = total_matrix + matrix

plt.matshow(total_matrix)
plt.colorbar()
plt.xlabel("groundTruth")
plt.ylabel("results")

plt.show()

get_numbers(total_matrix, labels)
