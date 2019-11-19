import os
import cv2
import numpy as np
from PIL import Image
from scipy.spatial.distance import pdist


def cosine(image1, image2):
    X = np.vstack([image1, image2])
    return pdist(X, 'cosine')[0]


def get_img(path):
    img_list = []
    counter = 0
    list_dir = sorted(os.listdir(path))
    print(len(list_dir))
    for item in list_dir:
        img_list.insert(counter, cv2.imread(
            path + '/' + str(item), cv2.IMREAD_GRAYSCALE))
        counter = counter + 1
    return img_list


def holder_clear(imglist):
    face_list = []
    counter = 0
    for item in imglist:
        face_list.insert(counter, item[165:290, 155:285])
        counter = counter + 1
    return face_list


def organize_img():
    return


if __name__ == "__main__":
    # 文件导入
    img_list1 = get_img('CT/Sample_1_20190818_131304')
    img_list2 = get_img('CT/Sample_1_20190818_134119')
    print(img_list2[1].shape)

    # 去容器
    img_list11 = holder_clear(img_list1)
    img_list22 = holder_clear(img_list2)

    cv2.imshow("demo1", img_list11[200])
    cv2.imshow("demo2", img_list22[2])

    # 亮度均衡

    # 相似度计算
    # print("余弦相似度：%f" % (cosine(image1, image2)))

    # 文件重排

    cv2.waitKey(0)
    cv2.destroyAllWindows()
