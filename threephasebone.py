import os
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
import pandas as pd
import cv2


def load_and_save(csv, root_path):

    # csv [ 0 -> bodypart; 1 -> type; 2 -> filename; 3 -> label]
    # type: 3 4, 4 4, 0 0
    info = np.loadtxt(csv, dtype=str, delimiter=",", skiprows=1)
    dirs = os.listdir(root_path)

    for i in range(len(info)):
        # 保存路径
        result_path = info[i, 0] + "/" + dirs[i]
        # 读取的文件路径
        file_path = os.path.join(root_path, dirs[i], info[i, 2])
        # 数据标签
        label_ = 1 if info[i, 3] == "1" else 0

        # 对图像处理, 按照灰度图进行读取
        # image = cv2.imread(file_path, cv2.IMREAD_COLOR)
        image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        flow_phase = np.zeros((12, 256, 256))
        pool_phase = np.zeros((4, 256, 256))
        # 保存裁剪出来的图片
        temp_cliped = []
        # 开始裁剪
        if info[i, 1] == "3 4":
            x = 130
            y = 77
            pic_width = 160
            pic_height = 120
            displacement_x = 406
            displacement_y = 169
        elif info[i, 1] == "4 4":
            x = 155
            y = 86 + 125
            pic_width = 110
            pic_height = 86
            displacement_x = 405
            displacement_y = 125
        else:
            continue

        for i in range(3):
            for j in range(4):
                per_image = image[
                    y + i * displacement_y : y + pic_height + i * displacement_y,
                    x + j * displacement_x : x + pic_width + j * displacement_x,
                ]
                # resize -> 256x256
                temp_cliped.append(per_image)
                per_image = cv2.resize(per_image, (256, 256))
                flow_phase[i * 4 + j] = per_image

        x = 294
        y = 602
        pic_width = 240
        pic_height = 180
        displacement_x = 813
        displacement_y = 253

        for i in range(2):
            for j in range(2):
                per_image = image[
                    y + i * displacement_y : y + pic_height + i * displacement_y,
                    x + j * displacement_x : x + pic_width + j * displacement_x,
                ]
                # resize -> 256x256
                temp_cliped.append(per_image)
                per_image = cv2.resize(per_image, (256, 256))
                pool_phase[i * 2 + j] = per_image

        # 保存文件和裁剪的图片
        for i in range(16):
            plt.subplot(4, 4, i + 1)
            plt.imshow(temp_cliped[i], plt.cm.gray)
            plt.axis("off")
        plt.savefig(result_path + ".png")
        np.savez(result_path + ".npz", flow=flow_phase, pool=pool_phase, label=label_)


if __name__ == "__main__":
    # load_and_save("ThreePhaseBone/total.csv", "ThreePhaseBone/2015-2021")
    img = sitk.ReadImage("ImageFileName.dcm")
    img__ = sitk.GetArrayFromImage(img)
    print(0)
