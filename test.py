import os
import cv2
import time
from datetime import datetime
import numpy as np
from PIL import Image
from scipy.spatial.distance import pdist
from matplotlib import pyplot as plt
import shapecontext
import shutil

now = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
path = "./output/"+str(now)

# 文件导入
dir1 = './CT/Sample_1_20190818_131304'
dir2 = './CT/Sample_1_20190818_134119'
dir3 = './CT/Sample_1_20190818_135700'
dir4 = './CT/Sample_1_20190818_141230'

# 图像处理


def get_img(path):
    img_list = []
    list_dir = sorted(os.listdir(path))
    # print(len(list_dir))
    for item in list_dir:
        img_list.append(cv2.imread(
            path + '/' + str(item), cv2.IMREAD_GRAYSCALE))
    return img_list


def holder_clear(img_list):
    face_list = []
    for item in img_list:
        face_list.append(item[165:290, 155:285])
    return face_list


def gaussian_blur(img_list):
    gaussian_list = []
    for item in img_list:
        gaussian_list.append(cv2.GaussianBlur(item, (3, 3), 0.5))
    return gaussian_list


def unsharping_mask(img_list):
    unsharping_list = []

    return unsharping_list


def binary_img(img_list):
    binary_list = []
    for item in img_list:
        binary_list.append(cv2.threshold(
            item, 20, 255, cv2.THRESH_BINARY)[1])
    return binary_list


def delete_min(img_list):
    delete_list = []
    for item in img_list:
        image, contours, hierarch = cv2.findContours(
            item, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for i in range(len(contours)):
            area = cv2.contourArea(contours[i])
            if area < 20:
                cv2.drawContours(image, [contours[i]], 0, 0, -1)
        delete_list.append(image)

    return delete_list


def delete_min2(img):
    delete_list = []
    img, contours, hierarch = cv2.findContours(
        img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        if area < 5:
            cv2.drawContours(img, [contours[i]], 0, 0, -1)
    return img


def canny_outline(img_list):
    canny_list = []
    for item in img_list:
        canny_list.append(cv2.Canny(item, 20, 80))
        # image, contours, hierarchy = cv2.findContours(
        #     item, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # cv2.drawContours(image, contours, 0, 0, -1)
        # canny_list.append(image)
        # print(image)
    return canny_list


def cosine(image1, image2):
    X = np.vstack([image1, image2])
    return pdist(X, 'cosine')[0]


def cal_similarity(img_list1, img_list2):
    res1 = []
    log = ""
    np.set_printoptions(suppress=True)
    counter = 0
    for index1, item1 in enumerate(img_list1):
        res2 = []
        for index2, item2 in enumerate(img_list2):
            # simi = float(cv2.matchShapes(
            #     item1, item2, cv2.CONTOURS_MATCH_I3, 0))

            # image1 = np.asarray(item1).flatten()
            # image2 = np.asarray(item2).flatten()
            # simi = cosine(image1,image2)
            # if simi == 0:
            #     simi = 100
            # print(str(index1)+"  "+str(index2))
            simi = shapecontext.shape_simi(item1, item2, 40)
            # 日志写入
            counter = counter + 1
            log = "（"+str(counter)+"/"+str(len(img_list1)*len(img_list2))+"）第"+str(index1) + "张和第"+str(index2) + \
                "张相似度为："+str(float(simi))+"\n"

            print(log)
            res2.append(simi)
            export_log(log)

        res1.append(res2)
    return res1


def cal_length(img_list1, img_list2):
    res1 = []
    for item1 in img_list1:
        _, contours1, hierarchy1 = cv2.findContours(
            item1, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        res2 = []
        for item2 in img_list2:
            _, contours2, hierarchy2 = cv2.findContours(
                item2, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            if len(contours2) == 0:
                res2.append(10000)
            else:
                res2.append(abs(cv2.arcLength(contours1[0], True) -
                                cv2.arcLength(contours2[0], True)))
        res1.append(res2)
    return res1


def inverse_color(img_list):
    inverse_list = []
    for item in img_list:

        height, width = item.shape
        img2 = item.copy()

        for i in range(height):
            for j in range(width):
                img2[i, j] = (255-item[i, j])

        inverse_list.append(img2)
    return inverse_list

# 文件操作


def export_log(log):
    with open(path+"/log_"+str(now)+".txt", "a") as f:
        f.write(log)


def export_img(img_list1, img_list2, file_name):
    sub_path = path + "/" + file_name
    sub_folder = os.path.exists(sub_path)

    # 文件夹创建
    if not sub_folder:
        os.makedirs(sub_path+"_1")
        os.makedirs(sub_path+"_2")
        print("---  new folder "+sub_path+"...  ---")
    else:
        print("---  There is this folder!  ---")

    counter = 0
    for item in img_list1:
        file_url = sub_path+"_1/"+str(counter)+".jpg"
        cv2.imwrite(file_url, item)
        counter = counter+1
    counter = 0
    for item in img_list2:
        file_url = sub_path+"_2/"+str(counter)+".jpg"
        cv2.imwrite(file_url, item)
        counter = counter+1


def organize_file(file_index1, file_index2):
    # 文件夹创建
    result_path = path + "/result"
    result_folder = os.path.exists(result_path)

    if not result_folder:
        os.makedirs(result_path)
        print("---  new folder "+result_path+"...  ---")
    else:
        print("---  There is this folder!  ---")

    index1 = file_index1
    index2 = file_index2
    old_file_list1 = os.listdir(dir1)
    old_file_list2 = os.listdir(dir2)
    while index1 != 0:
        try:
            shutil.copyfile(
                dir1 + '/' + old_file_list1[file_index1 - index1], result_path + '/' + str(file_index1 - index1)+'.png')
            # os.symlink(
            #     dir1 + '/' + old_file_list1[file_index1 - index1], result_path + '/' + str(file_index1 - index1)+'.png')
            index1 = index1 - 1
            print("正在导出："+result_path + '/' + str(file_index1 - index1)+'.png')
        except EOFError as e:
            index1 = index1 - 1
            print(e)

    while index2 != len(old_file_list2):
        try:
            shutil.copyfile(dir1 + '/' + old_file_list2[index1], result_path + '/' + str(
                file_index1 + file_index2 - index2+1)+'.png')
            # os.symlink(
            #     dir1 + '/' + old_file_list2[index1], result_path + '/' + str(file_index1 + file_index2 - index2+1)+'.png')
            index2 = index2 + 1
            print("正在导出："+result_path + '/' +
                  str(file_index1 + file_index2 - index2+1)+'.png')
        except EOFError as e:
            index2 = index2 + 1
            print(e)

    return


def main():
    # 创建日志
    folder = os.path.exists(path)
    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path)  # makedirs 创建文件时如果路径不存在会创建这个路径
        print("---  new folder "+path+"...  ---")
    else:
        print("---  There is this folder!  ---")

    import_start = time.time()
    img_list1 = get_img(dir1)
    img_list2 = get_img(dir2)
    import_end = time.time()
    print("导入时间："+str(import_end-import_start)+"秒")

    for i in range(240):
        img_list1.pop(0)
        img_list2.pop(len(img_list2)-1)

    # 去容器
    holder_start = time.time()
    img_holder_1 = holder_clear(img_list1)
    img_holder_2 = holder_clear(img_list2)
    holder_end = time.time()
    print("去容器时间："+str(holder_end-holder_start)+"秒")
    export_img(img_holder_1, img_holder_2, "1_holder")

    # 高斯滤波
    gaussian_start = time.time()
    img_gaussian1 = gaussian_blur(img_holder_1)
    img_gaussian2 = gaussian_blur(img_holder_2)
    gaussian_end = time.time()
    print("高斯滤波时间："+str(gaussian_end-gaussian_start)+"秒")
    export_img(img_gaussian1, img_gaussian2, "2_gaussian")

    # 二值化
    binary_start = time.time()
    img_binary1 = binary_img(img_gaussian1)
    img_binary2 = binary_img(img_gaussian2)
    binary_end = time.time()
    print("二值化时间："+str(binary_end-binary_start)+"秒")
    export_img(img_binary1, img_binary2, "3_binary")

    # 反色
    inverse_start = time.time()
    img_inverse1 = inverse_color(img_binary1)
    img_inverse2 = inverse_color(img_binary2)
    inverse_end = time.time()
    print("反色时间："+str(inverse_end-inverse_start)+"秒")
    export_img(img_inverse1, img_inverse2, "n_inverse")

    # 轮廓提取
    outline_start = time.time()
    img_outline1 = canny_outline(img_inverse1)
    img_outline2 = canny_outline(img_inverse2)
    outline_end = time.time()
    print("轮廓提取时间："+str(outline_end-outline_start)+"秒")
    export_img(img_outline1, img_outline2, "4_outline")

    # 去除多余部分
    img_final1 = delete_min(img_outline1)
    img_final2 = delete_min(img_outline2)
    # img_final1 = img_outline1
    # img_final2 = img_outline2

    # 相似度计算
    similarity_start = time.time()
    res = cal_similarity(img_final1, img_final2)
    # res = cal_length(img_final1, img_final2)
    similarity_end = time.time()
    print("相似度计算时间："+str(similarity_end-similarity_start)+"秒")

    # 文件导出
    export_start = time.time()
    export_img(img_final1, img_final2, "5_final")
    export_end = time.time()
    print("文件导出时间："+str(export_end-export_start)+"秒")

    # 求最小相似度
    min_list1 = []
    for item in res:
        min_list1.append(min(item))
    min1 = min_list1.index(min(min_list1))
    min2 = res[min1].index(min(res[min1]))
    index_log = "最相似的序号是"+str(240+min1)+"和"+str(min2)+"\n"
    simi_log = "相似度为："+str(res[min1][min2])+"\n"
    print(index_log)
    print(simi_log)
    export_log(index_log)
    export_log(simi_log)

    # print(cv2.matchShapes(img_final1[164], img_final2[16], cv2.CONTOURS_MATCH_I2, 1))
    shapecontext.shape_simi(img_inverse1[20], img_inverse2[16])

    # 绘图
    # titles = ["holder clean 1_"+str(min1+240), "holder clean 2_"+str(min2),
    #           "gaussian 1_"+str(min1+240), "gaussian 2_"+str(min2),
    #           "binary 1_"+str(min1+240), "binary 2_"+str(min2),
    #           "outline 1_"+str(min1+240), "outline 2_"+str(min2),
    #           "final 1_"+str(min1+240), "final 2_"+str(min2)]
    # img = [img_holder_1[min1], img_holder_2[min2],
    #        img_gaussian1[min1], img_gaussian2[min2],
    #        img_binary1[min1], img_binary2[min2],
    #        img_outline1[min1], img_outline2[min2],
    #        img_final1[min1], img_final2[min2]]

    # for i in range(10):
    #     plt.subplot(6, 2, i+1), plt.imshow(img[i], 'gray')
    #     plt.title(titles[i])
    #     plt.xticks([]), plt.yticks([])

    # plt.show()

    # 文件重排
    organize_file(min1 + 240, min2)


main()

# if __name__ == "__main__":
#     main()
