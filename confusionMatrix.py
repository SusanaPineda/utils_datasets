import numpy as np
import os
import cv2
import matplotlib.pyplot as plt

URL_ground_truth = "/home/susi/Documents/Datasets/data_9/val/2class_labels_YOLO/"
URL_results = "/home/susi/Documents/Pruebas_BG/Barcelona_240_5e05_0995_D2/"
URL_images = "/home/susi/Documents/Datasets/data_9/val/images/"

# labels = np.array(['Peaton_verde', 'Peaton_rojo', 'Peaton_generico', 'Coche_verde', 'Coche_rojo', 'Coche_generico'])
labels = np.array(['Peaton', 'Coche'])


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

    # im = cv2.resize(im, (int(im.shape[1]*0.2), int(im.shape[0]*0.2)))
    cv2.imshow("compare", im)
    cv2.waitKey(0)


def check_class(m_e, cl_r, cl_gt, d):
    m = np.zeros((len(cl_r), len(cl_gt)))
    for y in range(len(cl_gt)):
        for x in range(len(cl_r)):
            m[x][y] = cl_r[x] == cl_gt[y]
    m = np.logical_not(m)
    m = m * d / 2
    m_e = m_e + m
    return m_e


def update_matrix(sums, m_emp, cl_r, ps_r, cl_gt, ps_gt, idx_res, idx_gt):
    cl_gth = cl_gt[idx_gt]
    cl_res = cl_r[idx_res]
    sums[cl_res][cl_gth] = sums[cl_res][cl_gth] + 1
    m_emp[idx_res][idx_gt] = 1

    return sums, m_emp, cl_r, ps_r, cl_gt, ps_gt


def compare(cl_r, ps_r, cl_gt, ps_gt, d):
    sums = np.zeros((labels.size + 1, labels.size + 1))
    m_dist = np.zeros((len(cl_r), len(cl_gt)))
    m_emp = np.zeros((len(cl_r), len(cl_gt)))

    # for y, x in zip(range(len(ps_gt)), range(ps_r)):

    if len(cl_gt) > 0:
        for y in range(len(ps_gt)):
            for x in range(len(ps_r)):
                m_dist[x][y] = np.linalg.norm(np.asarray(ps_r[x]) - np.asarray(ps_gt[y]))

        if im_cls:
            m_dist = check_class(m_dist, cl_r, cl_gt, d)

        dis_min = np.argmin(m_dist, axis=1)
        # dis_min[i] = np.unravel_index(a, aux.shape)

        uniq, indx = np.unique(dis_min, return_index=True)
        if len(uniq) != len(cl_r):
            for i in range(len(uniq)):
                idx_num_det = np.where(dis_min == uniq[i])
                if len(idx_num_det) > 1:
                    distancias = np.zeros(len(idx_num_det))
                    for dis in range(len(idx_num_det)):
                        distancias[dis] = m_dist[uniq[i]][dis]
                    d_min = np.argmin(distancias)
                    dist = distancias[d_min]

                    if dist < d:
                        sums, m_emp, cl_r, ps_r, cl_gt, ps_gt = update_matrix(sums, m_emp, cl_r, ps_r, cl_gt, ps_gt,
                                                                              idx_num_det[d_min], uniq[i])

                else:
                    dist = m_dist[indx[i]][uniq[i]]
                    if dist < d:
                        sums, m_emp, cl_r, ps_r, cl_gt, ps_gt = update_matrix(sums, m_emp, cl_r, ps_r, cl_gt, ps_gt,
                                                                              indx[i], uniq[i])

        else:
            for i in range(len(dis_min)):
                dist = m_dist[i][dis_min[i]]
                if dist < d:
                    sums, m_emp, cl_r, ps_r, cl_gt, ps_gt = update_matrix(sums, m_emp, cl_r, ps_r, cl_gt, ps_gt,
                                                                          i, dis_min[i])

        fp = np.all(m_emp == 0, axis=1)
        for y in range(len(fp)):
            if fp[y]:
                sums[cl_r[y]][labels.size] = sums[cl_r[y]][labels.size] + 1

        fn = np.all(m_emp == 0, axis=0)
        for x in range(len(fn)):
            if fn[x]:
                sums[labels.size][cl_gt[x]] = sums[labels.size][cl_gt[x]] + 1

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
total_matrix = np.zeros((labels.size + 1, labels.size + 1))
vis = False
im_cls = True

for d in data:

    results = open(os.path.join(URL_results, d))
    gt = open(os.path.join(URL_ground_truth, d))

    img = cv2.imread(os.path.join(URL_images, d.split('.')[0] + ".png"))

    distance = img.shape[0] / 5

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
