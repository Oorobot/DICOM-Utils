import os
from datetime import datetime
from glob import glob
from re import M

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dicom import get_pixel_value


# 处理骨三相 IMG 等图片格式文件
def img_process(filename: str, crop_type: str, label: int, save_path: str):

    if crop_type == "3 4":
        x = 130
        y = 77
        pic_width = 160
        pic_height = 120
        displacement_x = 406
        displacement_y = 169
    elif crop_type == "4 4":
        x = 155
        y = 86 + 125
        pic_width = 110
        pic_height = 86
        displacement_x = 405
        displacement_y = 125
    else:
        return

    # 读取图像(灰度图)
    image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    # 裁剪下来的16张图象, blood flow phase 前12张, blood pool phase 后4张
    cropped_images = []
    # 每张裁剪下来后resize到256x256的图像
    resized_images = []

    for i in range(3):
        for j in range(4):
            an_image = image[
                y + i * displacement_y : y + pic_height + i * displacement_y,
                x + j * displacement_x : x + pic_width + j * displacement_x,
            ]
            cropped_images.append(an_image)
            # resize -> 256x256
            resized_images.append(cv2.resize(an_image, (256, 256)))

    x = 294
    y = 602
    pic_width = 240
    pic_height = 180
    displacement_x = 813
    displacement_y = 253

    for i in range(2):
        for j in range(2):
            an_image = image[
                y + i * displacement_y : y + pic_height + i * displacement_y,
                x + j * displacement_x : x + pic_width + j * displacement_x,
            ]
            # resize -> 256x256
            cropped_images.append(an_image)
            # resize -> 256x256
            resized_images.append(cv2.resize(an_image, (256, 256)))

    resized_images = np.array(resized_images)
    # 保存裁剪的图片
    for i in range(16):
        plt.subplot(4, 4, i + 1)
        plt.imshow(cropped_images[i], plt.cm.gray)
        plt.axis("off")
    plt.savefig(save_path + "_JPG_CROP.png")
    plt.close()
    # 保存resize后的图片
    for i in range(16):
        plt.subplot(4, 4, i + 1)
        plt.imshow(resized_images[i], plt.cm.gray)
        plt.axis("off")
    plt.savefig(save_path + "_JPG_RESIZE.png")
    plt.close()
    # 保存数据文件
    np.savez(save_path + "_JPG.npz", data=resized_images, label=label)


# 处理骨三相 DCM 格式文件
def dcm_process(
    pixel_value: np.ndarray, label: int, save_path: str, mask: np.ndarray = None
):
    flow_and_pool = pixel_value[0:25]
    # 保存图片
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.imshow(flow_and_pool[i], plt.cm.binary)
        plt.axis("off")
    plt.savefig(save_path + "_DCM.png")
    plt.close()
    # 保存数据文件
    if mask is None:
        np.savez(save_path + "_DCM.npz", data=flow_and_pool, label=label)
    else:
        imgs = [
            flow_and_pool[24],
            mask[24],
            np.multiply(flow_and_pool[24], 1 - mask[24]),
        ]
        for i in range(3):
            plt.subplot(1, 3, i + 1)
            plt.imshow(imgs[i], plt.cm.binary)
            plt.axis("off")
        plt.savefig(save_path + "_DCM_.png")
        plt.close()
        np.savez(save_path + "_DCM.npz", data=flow_and_pool, label=label, mask=mask[24])


# 读取数据信息
xlsx = pd.read_excel("ThreePhaseBone/ThreePhaseBone.xlsx")
infos = xlsx[["编号", "最终结果", "部位", "type"]].values

# # JPG数据处理
# jpgs = glob("ThreePhaseBone/*/*_1.JPG")
# for jpg in jpgs:
#     index = jpg.split("\\")[-1].split("_")[0]
#     info = infos[int(index) - 1]
#     bodypart = "hip" if info[2] == "髋" else "knee"
#     save_path = os.path.join("ProcessedData", bodypart, index)
#     img_process(jpg, info[-1], info[1], save_path)

# DCM数据处理
# dcms = glob("ThreePhaseBone/*/*/*_FLOW.dcm")
# for d in dcms:
#     index = d.split("\\")[-1].split("_")[0]
#     info = infos[int(index) - 1]
#     bodypart = "hip" if info[2] == "髋" else "knee"
#     save_path = os.path.join("ProcessedData", bodypart, index)
#     pixel_value = get_pixel_value(d)
#     dcm_process(pixel_value, info[1], save_path)

# 带有标注数据的 Hip 数据处理
dcms = glob("ThreePhaseBone/hip_/*/*_FLOW.dcm")
masks = glob("ThreePhaseBone/hip_/*/*.nii.gz")
for d, m in zip(dcms, masks):
    index = d.split("\\")[-1].split("_")[0]
    info = infos[int(index) - 1]
    save_path = os.path.join("ProcessedData", "hip_", index)
    pixel_value = get_pixel_value(d)
    mask_value = get_pixel_value(m)
    dcm_process(pixel_value, info[1], save_path, mask_value)
