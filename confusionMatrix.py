import numpy as np
import os
import cv2

URL_ground_truth = "../Datasets/validacion_semaforos/labels/"
URL_results = "../Datasets/validacion_semaforos/labels_results/"
URL_images = "../Datasets/validacion_semaforos/images/"


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
            cn = (left+width/2, top+height/2)

        cl.append(tag)
        center.append(cn)

    return cl, center


def compare(cl_r, ps_r, cl_gt, ps_gt):
    class_results_copy = cl_r.copy()
    class_gt_copy = cl_gt.copy()
    cont = 0

    sums = np.zeros((8, 8))
    if len(cl_r) > 0:
        for i in range(len(cl_gt)):
            np_pos = np.asarray(ps_gt[i])
            diff = np.absolute(np.subtract(ps_r, np_pos))
            dis_min = np.where(diff == np.amin(diff))

            cl_gth = cl_gt[i]

            if (len(class_gt_copy) > 0) & (len(class_results_copy) > 0):
                if dis_min[0].size == 1:
                    cl_res = cl_r[dis_min[0][0]]
                    dist = np.linalg.norm(np.asarray(ps_r[dis_min[0][0]]) - np_pos)
                    if dist < 90:
                        sums[cl_res][cl_gth] = sums[cl_res][cl_gth] + 1
                        class_results_copy = np.delete(class_results_copy, dis_min[0][0] - cont)
                        class_gt_copy = np.delete(class_gt_copy, i - cont)
                        cont = cont + 1
                else:
                    cl_res = cl_r[dis_min[0][1]]
                    dist = np.linalg.norm(np.asarray(ps_r[dis_min[0][1]]) - np_pos)
                    if dist < 90:
                        sums[cl_res][cl_gth] = sums[cl_res][cl_gth] + 1
                        class_results_copy = np.delete(class_results_copy, dis_min[0][1] - cont)
                        class_gt_copy = np.delete(class_gt_copy, i - cont)
                        cont = cont + 1

    while len(class_gt_copy) > 0:
        sums[7][class_gt_copy[0]] = sums[7][class_gt_copy[0]] + 1
        class_gt_copy = np.delete(class_gt_copy, 0)

    while len(class_results_copy) > 0:
        sums[class_results_copy[0]][6] = sums[class_results_copy[0]][6] + 1
        class_results_copy = np.delete(class_results_copy, 0)

    return sums


data = os.listdir(URL_ground_truth)
total_matrix = np.zeros((8, 8))

for d in data:
    results = open(os.path.join(URL_results, d))
    gt = open(os.path.join(URL_ground_truth, d))

    if len(d.split('.')[0].split('_')) == 3:
        img = cv2.imread(os.path.join(URL_images, d.split('.')[0] + ".jpg"))
    else:
        img = cv2.imread(os.path.join(URL_images, d.split('.')[0] + ".png"))

    class_results, positions_results = get_info(results, img, "results")
    class_gt, positions_gt = get_info(gt, img, "gt")

    matrix = compare(class_results, positions_results, class_gt, positions_gt)

    total_matrix = total_matrix + matrix
print(total_matrix)