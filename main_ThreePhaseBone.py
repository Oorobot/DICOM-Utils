import os
from datetime import datetime
from glob import glob
from re import M
from turtle import right

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dicom import get_pixel_value
from utils import mkdir


def get_mask_boundary(contour):
    contour = np.squeeze(contour)
    right, lower = np.max(contour, axis=0)
    left, upper = np.min(contour, axis=0)
    return upper, lower, left, right


def save_four_images(imgs, titles, camps, path):
    for i in range(4):
        plt.subplot(2, 2, i + 1)
        plt.title(titles[i])
        plt.imshow(imgs[i], camps[i])
        plt.axis("off")
    plt.savefig(path)
    plt.close()


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
    pixel_value: np.ndarray, label: int, save_path: str, mask: np.ndarray = None,
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
# xlsx = pd.read_excel("ThreePhaseBone/ThreePhaseBone.xlsx")
# infos = xlsx[["编号", "最终结果", "部位", "type"]].values

# # JPG数据处理
# jpgs = glob("ThreePhaseBone/*/*_1.JPG")
# for jpg in jpgs:
#     index = jpg.split("\\")[-1].split("_")[0]
#     info = infos[int(index) - 1]
#     bodypart = "hip" if info[2] == "髋" else "knee"
#     save_path = os.path.join("_ProcessedData_", bodypart, index)
#     img_process(jpg, info[-1], info[1], save_path)

# DCM数据处理
# dcms = glob("ThreePhaseBone/*/*/*_FLOW.dcm")
# for d in dcms:
#     index = d.split("\\")[-1].split("_")[0]
#     info = infos[int(index) - 1]
#     bodypart = "hip" if info[2] == "髋" else "knee"
#     save_path = os.path.join("_ProcessedData_", bodypart, index)
#     pixel_value = get_pixel_value(d)
#     dcm_process(pixel_value, info[1], save_path)

# 带有标注无关区域数据的 Hip 数据处理
# dcms = glob("ThreePhaseBone/hip_none/*/*_FLOW.dcm")
# masks = glob("ThreePhaseBone/hip_none/*/*.nii.gz")
# for d, m in zip(dcms, masks):
#     index = d.split("\\")[-1].split("_")[0]
#     info = infos[int(index) - 1]
#     save_path = os.path.join("_ProcessedData_", "hip_", index)
#     pixel_value = get_pixel_value(d)
#     mask_value = get_pixel_value(m)
#     dcm_process(pixel_value, info[1], save_path, mask_value)


# 带有标注髋部区域数据的 Hip 数据处理
# label: 0 -> 正常, 1 -> 置换手术后非感染, 2 -> 置换手术后感染


xlsx = pd.read_excel("ThreePhaseBone/hip_focus/hip_focus.xlsx")
infos = xlsx[["编号", "最终结果", "左右"]].values
infos_index = 0

dcms = glob("ThreePhaseBone/hip_focus/*/*FLOW.dcm")
masks = glob("ThreePhaseBone/hip_focus/*/*.nii.gz")
save_dir = os.path.join("_ProcessedData_", "hip_focus")
labels = []
mkdir(save_dir)

normal_left_hip = []
normal_right_hip = []
sum_left_hip, sum_right_hip = (
    np.zeros((25, 40, 40), dtype=np.float32),
    np.zeros((25, 40, 40), dtype=np.float32),
)

for d, m in zip(dcms, masks):
    d_dir = os.path.dirname(d)
    m_dir = os.path.dirname(m)
    print(f"===> 正在处理 {d_dir} 文件夹下的文件")
    if d_dir != m_dir:
        print(f"====> 骨三相文件与标签文件不匹配 {d_dir} != {m_dir}, 跳过!")
        continue
    index = int(m_dir.split("\\")[-1])

    # 找到对应数据的信息
    while True:
        if infos[infos_index][0] == index:
            print(f"=====> 找到当前 {d_dir} 文件夹下的文件信息 {infos[infos_index]}")
            break
        print(f"====> {infos[infos_index]} 编号不匹配.")
        infos_index += 1

    info = infos[infos_index]
    # 本次文件信息已提取，自动 infos_index + 1
    infos_index += 1
    d_value = get_pixel_value(d)
    m_value = get_pixel_value(m)[24]
    contours, hierarchy = cv2.findContours(
        m_value.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE,
    )
    if len(contours) != 2:
        print(f"====> 当前文件没有 2 个分割轮廓, 跳过.")
        continue

    upper1, lower1, left1, right1 = get_mask_boundary(np.squeeze(contours[0]))
    upper2, lower2, left2, right2 = get_mask_boundary(np.squeeze(contours[1]))

    # first_right = True, contour[0] -> R, contour[1] -> L
    # first_right = False, contour[0] -> L, contour[1] -> R
    first_right = True if left1 < left2 else False

    # label: 0 -> 正常, 1 -> 置换手术后非感染, 2 -> 置换手术后感染
    r_label, l_label = -1, -1
    if info[1] == 0:  # 非感染
        if info[2] == "L":  # L 侧置换手术
            r_label, l_label = 0, 1
        elif info[2] == "R":  # R 侧置换手术
            r_label, l_label = 1, 0
        elif info[2] == "D":  # 双侧置换手术
            r_label, l_label = 1, 1
    elif info[1] == 1:  # 感染
        if info[2] == "L":  # L 侧置换手术
            r_label, l_label = 0, 2
        elif info[2] == "R":  # R 侧置换手术
            r_label, l_label = 2, 0
        elif info[2] == "D":  # 双侧置换手术
            r_label, l_label = 2, 2
        elif info[2] == "R感染L非感染":  # 双侧置换手术
            r_label, l_label = 2, 1
        elif info[2] == "L感染R非感染":  # 双侧置换手术
            r_label, l_label = 1, 2
    if l_label == -1 or r_label == -1:
        print(f"====> 当前文件找不到对应标签, 跳过.")
        continue

    labels.extend([r_label, l_label])

    if r_label == 0:
        normal_right_hip.append(
            d_value[0:25, upper1 : lower1 + 1, left1 : right1 + 1]
            if first_right
            else d_value[0:25, upper2 : lower2 + 1, left2 : right2 + 1]
        )
        sum_right_hip += (
            d_value[0:25, upper1 : lower1 + 1, left1 : right1 + 1]
            if first_right
            else d_value[0:25, upper2 : lower2 + 1, left2 : right2 + 1]
        )
    if l_label == 0:
        normal_left_hip.append(
            d_value[0:25, upper2 : lower2 + 1, left2 : right2 + 1]
            if first_right
            else d_value[0:25, upper1 : lower1 + 1, left1 : right1 + 1]
        )
        sum_left_hip += (
            d_value[0:25, upper2 : lower2 + 1, left2 : right2 + 1]
            if first_right
            else d_value[0:25, upper1 : lower1 + 1, left1 : right1 + 1]
        )
    # imgs = [
    #     d_value[24],
    #     m_value,
    # ]
    # r_hip_filename = os.path.join(save_dir, str(index).zfill(3) + f"_r_{r_label}.npz")
    # l_hip_filename = os.path.join(save_dir, str(index).zfill(3) + f"_l_{l_label}.npz")
    # # if first_right:
    # np.savez(
    #     r_hip_filename if first_right else l_hip_filename,
    #     data=d_value[0:25],
    #     label=r_label if first_right else l_label,
    #     boundary=[upper1, lower1, left1, right1],
    # )
    # np.savez(
    #     l_hip_filename if first_right else r_hip_filename,
    #     data=d_value[0:25],
    #     label=l_label if first_right else r_label,
    #     boundary=[upper2, lower2, left2, right2],
    # )

    # imgs.extend(
    #     [
    #         d_value[24, upper1 : lower1 + 1, left1 : right1 + 1],
    #         d_value[24, upper2 : lower2 + 1, left2 : right2 + 1],
    #     ]
    #     if first_right
    #     else [
    #         d_value[24, upper2 : lower2 + 1, left2 : right2 + 1],
    #         d_value[24, upper1 : lower1 + 1, left1 : right1 + 1],
    #     ]
    # )
    # save_four_images(
    #     imgs,
    #     titles=["hip", "mask", f"right {r_label}", f"left {l_label}"],
    #     camps=[plt.cm.binary, plt.cm.binary, plt.cm.binary, plt.cm.binary],
    #     path=os.path.join(save_dir, str(index).zfill(3) + ".png"),
    # )

print("各类标签样本数量: ", np.bincount(labels))
right_hip = sum_right_hip / len(normal_right_hip)
left_hip = sum_left_hip / len(normal_left_hip)
np.savez("normal_hip.npz", right=right_hip, left=left_hip)
print(0)


# 获取 左髋 和 右髋 （正常） 的平均值

