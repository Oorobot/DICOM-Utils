import os
from copy import deepcopy
from glob import glob
from types import TracebackType
from typing import List

import cv2
import numpy as np
import pandas as pd

from utils.dicom import get_pixel_value
from utils.utils import OUTPUT_FOLDER, mkdirs


def normalize(
    input: np.ndarray,
    mask: np.ndarray = None,
    dicom_window_ratio: float = 0.5,
    reduction_ratio: float = 1.0 / 6.0,
):
    if input.shape[0] == 25:
        flow_max = np.max(input[00:20]) * dicom_window_ratio
        pool_max = np.max(input[20:25]) * dicom_window_ratio
        if mask is not None:
            input = input * (1 - mask) + input * (mask * reduction_ratio)
        normalized_flow = np.clip(input[00:20], 0, flow_max) / flow_max
        normalized_pool = np.clip(input[20:25], 0, pool_max) / pool_max
        return np.concatenate((normalized_flow, normalized_pool), axis=0)
    else:
        max = np.max(input) * dicom_window_ratio
        if mask is not None:
            input = input * (1 - mask) + input * (mask * reduction_ratio)
        normalized_input = np.clip(input, 0, max) / max
        return normalized_input


def get_mask_boundary(contour):
    contour = np.squeeze(contour)
    right, lower = np.max(contour, axis=0)
    left, upper = np.min(contour, axis=0)
    return upper, lower, left, right


# 处理骨三相 IMG 等图片格式文件
def image_process(filelist: List[str], infomation: list, data_folders):
    for file in filelist:
        index = os.path.basename(os.path.dirname(file))
        info = infomation[int(index) - 1]
        save_folder = data_folders[1] if info[2] == "髋" else data_folders[0]
        crop_type = info[-1]
        label = info[1]

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
            continue

        # 读取图像(灰度图)
        image = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
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
        # 保存数据文件
        np.savez(
            os.path.join(save_folder, f"JPG_{index}"),
            data=resized_images,
            label=label,
        )
        # 保存图片
        for i in range(16):
            cv2.imwrite(
                os.path.join(save_folder, "images", f"JPG_{index}_{i:0>2}.jpg"),
                resized_images[i],
            )


# 处理骨三相 DICOM 格式文件
def DICOM_process(
    filelist: List[str],
    infomation: list,
    data_folders: List[str],
    use_mask: bool = False,
):
    for file in filelist:
        index = os.path.basename(os.path.dirname(file))
        info = infomation[int(index) - 1]
        label = info[1]
        pixel_value = get_pixel_value(file)
        flow_pool = pixel_value[0:25]

        flow_pool_ = deepcopy(flow_pool)
        images = 255 - normalize(flow_pool_) * 255

        # 判断是否存在mask, 进行相应的处理操作
        if use_mask:
            # 是否存在对应的 mask
            mask_file = os.path.join(os.path.dirname(file), "mask.nii.gz")
            if not os.path.exists(mask_file):
                continue
            # 获取 mask 进行相应操作
            mask = get_pixel_value(mask_file)[24]
            # 保存数据文件
            save_folder = data_folders[2]
            np.savez(
                os.path.join(save_folder, f"DCM_{index}"),
                data=flow_pool,
                label=label,
            )
            # 绘制 mask 的轮廓到原图像中，仅绘制血池相的最后一张
            contours, hierarchy = cv2.findContours(
                mask.astype(np.uint8),
                cv2.RETR_LIST,
                cv2.CHAIN_APPROX_NONE,
            )
            masked_image = deepcopy(images[24])
            masked_image = np.repeat(masked_image[:, :, np.newaxis], 3, 2)
            cv2.drawContours(masked_image, contours, -1, (0, 0, 255), 1)
            cv2.imwrite(
                os.path.join(save_folder, "images", f"DCM_{index}_mask.jpg"),
                masked_image,
            )
        else:
            save_folder = data_folders[1] if info[2] == "髋" else data_folders[0]
            np.savez(
                os.path.join(save_folder, f"DCM_{index}"), data=flow_pool, label=label
            )
        # 保存图片
        for i in range(25):
            cv2.imwrite(
                os.path.join(save_folder, "images", f"DCM_{index}_{i:0>2}.jpg"),
                images[i],
            )


def DICOM_process_with_ROI(roi_filelist: List[str], infomation: list, data_folders):
    save_folder = data_folders[3]
    labels = []
    sum_normal_left, sum_normal_right = (
        np.zeros((25, 40, 40), dtype=np.float32),
        np.zeros((25, 40, 40), dtype=np.float32),
    )
    num_normal_left, num_normal_right = 0, 0
    i = -1
    for roi in roi_filelist:
        i += 1
        info = infomation[i]
        folder_name = os.path.dirname(roi)
        index = os.path.basename(folder_name)
        while int(index) != info[0]:
            i += 1
            info = infomation[i]
        file = os.path.join(folder_name, f"{index}_FLOW.dcm")
        print(f"===> 正在处理 {folder_name} 文件夹下的文件")
        f = get_pixel_value(file)
        r = get_pixel_value(roi)[24]
        contours, hierarchy = cv2.findContours(
            r.astype(np.uint8),
            cv2.RETR_LIST,
            cv2.CHAIN_APPROX_NONE,
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
        labels += [l_label, r_label]
        if r_label == 0:
            sum_normal_right += (
                f[0:25, upper1 : lower1 + 1, left1 : right1 + 1]
                if first_right
                else f[0:25, upper2 : lower2 + 1, left2 : right2 + 1]
            )
            num_normal_right += 1
        if l_label == 0:
            sum_normal_left += (
                f[0:25, upper2 : lower2 + 1, left2 : right2 + 1]
                if first_right
                else f[0:25, upper1 : lower1 + 1, left1 : right1 + 1]
            )
            num_normal_left += 1
        r_hip_filename = os.path.join(save_folder, f"{str(index).zfill(3)}_r_{r_label}")
        l_hip_filename = os.path.join(save_folder, f"{str(index).zfill(3)}_l_{l_label}")
        np.savez(
            r_hip_filename if first_right else l_hip_filename,
            data=f[0:25],
            label=r_label if first_right else l_label,
            boundary=[upper1, lower1, left1, right1],
        )
        np.savez(
            l_hip_filename if first_right else r_hip_filename,
            data=f[0:25],
            label=l_label if first_right else r_label,
            boundary=[upper2, lower2, left2, right2],
        )
        # 保存图像
        images = deepcopy(f)
        images = 255 - normalize(images[0:25]) * 255
        image = images[24].astype(np.uint8)
        image = np.repeat(image[:, :, np.newaxis], repeats=3, axis=2)
        # image = image.astype(np.uint8)
        cv2.rectangle(image, (left1, upper1), (right1, lower1), (0, 0, 255), 1)
        cv2.rectangle(image, (left2, upper2), (right2, lower2), (0, 0, 255), 1)
        cv2.imwrite(
            os.path.join(save_folder, "images", f"{index}_l_{l_label}_r_{r_label}.jpg"),
            image,
        )

    print("各类标签样本数量: ", np.bincount(labels))
    left_hip = sum_normal_left / num_normal_left
    right_hip = sum_normal_right / num_normal_right
    # 保存正常左髋和右髋的平均值
    np.savez(os.path.join(save_folder, "normal_hip"), right=right_hip, left=left_hip)
    print("结束!")


# 创建文件夹
base_folder = os.path.join(OUTPUT_FOLDER, "ThreePhaseBone")
data_folders = [
    os.path.join(base_folder, d) for d in ["knee", "hip", "hip_mask", "hip_roi"]
]
image_folders = [os.path.join(d, "images") for d in data_folders]
mkdirs([OUTPUT_FOLDER, base_folder] + data_folders + image_folders)

# 读取数据信息
ThreePhaseBone = pd.read_excel("ThreePhaseBone/ThreePhaseBone.xlsx")
TB_info = ThreePhaseBone[["编号", "最终结果", "部位", "type"]].values

JPG_files = glob("ThreePhaseBone/*/*/*_1.JPG")
image_process(JPG_files, TB_info, data_folders)

DICOM_files = glob("ThreePhaseBone/*/*/*_FLOW.dcm")
DICOM_process(DICOM_files, TB_info, data_folders)

MASK_files = glob("ThreePhaseBone/hip/*/mask.nii.gz")
DICOM_process(DICOM_files, TB_info, data_folders, True)

# 带有标注髋部区域数据的 Hip 数据处理
# label: 0 -> 正常, 1 -> 置换手术后非感染, 2 -> 置换手术后感染
# Hip_ROI = pd.read_excel("ThreePhaseBone/hip_roi.xlsx")
# Hip_ROI_info = Hip_ROI[["编号", "最终结果", "左右"]].values
# ROI_files = glob("ThreePhaseBone/hip/*/roi.nii.gz")
# DICOM_process_with_ROI(ROI_files, Hip_ROI_info, data_folders)
