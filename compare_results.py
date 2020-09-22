import os
import cv2

input_URL_images = "/home/susi/Documents/Datasets/data_8/val2/images/"
input_URL_results_labels = "/home/susi/Documents/Pruebas_BG/Barcelona_P1_Val2/"
input_URL_gt_labels = "/home/susi/Documents/Datasets/data_8/val2/labels/"
output_URL = "/home/susi/Documents/Datasets/Barcelona_val2/"

colors_gt = [(8, 186, 255), (7, 163, 250), (4, 93, 232), (2, 47, 220), (0, 0, 208), (8, 2, 157)]
colors_results = [(244, 232, 173), (228, 202, 72), (216, 180, 0), (182, 119, 0), (138, 62, 2), (94, 4, 3)]


def paint(file, img, t):
    text = file.read()
    text = text.split()
    h = img.shape[0]
    w = img.shape[1]
    for i in range(int(len(text) / 5)):
        despl = i * 5
        tag = int(text[0 + despl])

        if t == "gt":
            left = (float(text[1 + despl]) * w) - (float(text[3 + despl]) * w / 2)
            top = (float(text[2 + despl]) * h) - (float(text[4 + despl]) * h / 2)
            right = (float(text[1 + despl]) * w) + (float(text[3 + despl]) * w / 2)
            bottom = (float(text[2 + despl]) * h) + (float(text[4 + despl]) * h / 2)

            img = cv2.rectangle(img, (int(left), int(top)), (int(right), int(bottom)), colors_gt[tag], 2)
        else:
            left = float(text[1 + despl])
            top = float(text[2 + despl])
            right = float(text[3 + despl])
            bottom = float(text[4 + despl])

            img = cv2.rectangle(img, (int(left), int(top)), (int(right), int(bottom)), colors_results[tag], 2)

    return img


data = os.listdir(input_URL_gt_labels)
save = True
visualize = False

for d in data:
    file = open(os.path.join(input_URL_results_labels, d))
    file_gt = open(os.path.join(input_URL_gt_labels, d))
    print(file)

    img = cv2.imread(os.path.join(input_URL_images, d.split('.')[0] + ".png"))
    if img is None:
        img = cv2.imread(os.path.join(input_URL_images, d.split('.')[0] + ".jpg"))

    img = paint(file_gt, img, "gt")
    img = paint(file, img, "results")

    if visualize:
        cv2.imshow("img", img)
        cv2.waitKey(0)



    if save:
        cv2.imwrite(os.path.join(output_URL, d.split('.')[0] + ".png"), img)
