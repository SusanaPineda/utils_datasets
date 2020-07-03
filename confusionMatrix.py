import numpy as np
import os
import cv2
import matplotlib.pyplot as plt

URL_ground_truth = "../Datasets/validacion_semaforos/labels_2/"
URL_results = "../Datasets/validacion_semaforos/results_firstDataset_lastNet/"
URL_images = "../Datasets/validacion_semaforos/images_2/"


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
            cn = (top+height/2, left+width/2)

        cl.append(tag)
        center.append(cn)

    return cl, center

def paint_detections(pr, pgt, im):
    for p in pr:
        cv2.circle(im, (int(p[0]), int(p[1])), 3, (255, 0, 0), 2)
    for g in pgt:
        cv2.circle(im, (int(g[0]), int(g[1])), 3, (0, 0, 255), 2)

    cv2.imshow("compare", im)
    cv2.waitKey(0)

def compare(cl_r, ps_r, cl_gt, ps_gt):
    i = 0
    cont = len(cl_gt)
    sums = np.zeros((8, 8))

    while (len(cl_r) > 0) & (len(cl_gt) > 0) & (cont > 0):
        cont = cont-1
        np_pos = np.asarray(ps_gt[i])
        #diff = np.absolute(np.subtract(ps_r, np_pos))
        diff = [np.linalg.norm(np.asarray(x) - np_pos) for x in list(ps_r)]
        dis_min = np.argmin(diff)
        #dis_min = int(dis_min/2)

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
            i = i+1

    while len(cl_gt) > 0:
        sums[7][cl_gt[0]] = sums[7][cl_gt[0]] + 1
        cl_gt = np.delete(cl_gt, 0)

    while len(cl_r) > 0:
        sums[cl_r[0]][6] = sums[cl_r[0]][6] + 1
        cl_r = np.delete(cl_r, 0)

    return sums


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
        paint_detections(positions_results, positions_gt, img.copy())

    matrix = compare(class_results.copy(), positions_results.copy(), class_gt.copy(), positions_gt.copy())

    total_matrix = total_matrix + matrix

plt.matshow(total_matrix)
plt.colorbar()
plt.xlabel("groundTruth")
plt.ylabel("results")
plt.show()

