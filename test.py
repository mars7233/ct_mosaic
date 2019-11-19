import os
import cv2
import numpy as np
from PIL import Image
from scipy.spatial.distance import pdist
from matplotlib import pyplot as plt
import time
from scipy._lib.six import xrange


def cosine(image1, image2):
    X = np.vstack([image1, image2])
    return pdist(X, 'cosine')[0]


def get_img(path):
    img_list = []
    counter = 0
    list_dir = sorted(os.listdir(path))
    # print(len(list_dir))
    for item in list_dir:
        img_list.insert(counter, cv2.imread(
            path + '/' + str(item), cv2.IMREAD_GRAYSCALE))
        counter = counter + 1
    return img_list


def holder_clear(img_list):
    face_list = []
    counter = 0
    for item in img_list:
        face_list.insert(counter, item[165:290, 155:285])
        counter = counter + 1
    return face_list


def gaussian_blur(img_list):
    gaussian_list = []
    counter = 0
    for item in img_list:
        gaussian_list.insert(counter, cv2.GaussianBlur(item, (3, 3), 0.5))
        counter = counter + 1
    return gaussian_list


def unsharping_mask(img_list):
    unsharping_list = []

    return unsharping_list


def binary_img(img_list):
    binary_list = []
    counter = 0
    for item in img_list:
        binary_list.insert(counter, cv2.threshold(
            item, 30, 200, cv2.THRESH_BINARY)[1])
        counter = counter + 1
    return binary_list


def canny_outline(img_list):
    canny_list = []
    counter = 0
    for item in img_list:
        canny_list.insert(counter, cv2.Canny(item, 100, 200))
        counter = counter + 1
    return canny_list


def organize_img():
    return


def main():
    # 文件导入
    import_start = time.time()
    img_list1 = get_img('CT/Sample_1_20190818_131304')
    img_list2 = get_img('CT/Sample_1_20190818_134119')
    import_end = time.time()
    print("导入时间："+str(import_end-import_start)+"秒")
    # print(img_list2[1].shape)

    # 去容器
    holder_start = time.time()
    img_holder_1 = holder_clear(img_list1)
    img_holder_2 = holder_clear(img_list2)
    holder_end = time.time()
    print("去容器时间："+str(holder_end-holder_start)+"秒")

    # 高斯滤波
    gaussian_start = time.time()
    img_gaussian1 = gaussian_blur(img_holder_1)
    img_gaussian2 = gaussian_blur(img_holder_2)
    gaussian_end = time.time()
    print("高斯滤波时间："+str(gaussian_end-gaussian_start)+"秒")

    # 二值化
    binary_start = time.time()
    img_binary1 = binary_img(img_gaussian1)
    img_binary2 = binary_img(img_gaussian2)
    binary_end = time.time()
    print("二值化时间："+str(binary_end-binary_start)+"秒")

    # mars = cv2.Canny(img_gaussian2[2], 100, 200)

    # 轮廓提取
    outline_start = time.time()
    img_outline1 = canny_outline(img_binary1)
    img_outline2 = canny_outline(img_binary2)
    outline_end = time.time()
    print("轮廓提取时间："+str(outline_end-outline_start)+"秒")

    # 绘图
    titles = ["1", "2"]
    img = [img_outline1[250], img_outline2[2]]
    for i in xrange(2):
        plt.subplot(1, 2, i+1), plt.imshow(img[i], 'gray')
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])
    plt.show()

    # 相似度计算
    # print("余弦相似度：%f" % (cosine(image1, image2)))

    # 文件重排

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
