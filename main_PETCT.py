import copy
import os
from glob import glob
from math import sqrt
from typing import List, Tuple

import cv2
import numpy as np
import pandas as pd
import SimpleITK as sitk

from utils.dicom import (
    ct2image,
    get_SUV_in_GE,
    read_serises_image,
    resample_based_target_image,
)
from utils.utils import OUTPUT_FOLDER, load_json, mkdirs, save_json


def crop_based_lesions(
    boundary: Tuple[int, int, int, int],  # [left, upper, right, lower]
    cliped_size: Tuple[int, int] = (32, 32),  # [width, height]
    pic_size: Tuple[int, int] = (512, 512),  # [width, height]
):
    """以病灶为中心剪切出 cliped size 的图片"""

    left, upper, right, lower = boundary
    # boundary 左右均为闭区间, 即[left, right], [upper, lower].
    boundary_width = right - left
    boundary_height = lower - upper
    # boundary's center
    center_x, center_y = (
        (left + right) * 0.5,
        (upper + lower) * 0.5,
    )

    def get_boundary(center, cliped_length, pic_length):
        if center - cliped_length * 0.5 <= 0:
            min, max = 0, cliped_length + 1
        elif center + cliped_length * 0.5 >= pic_length:
            min, max = pic_length - cliped_length - 1, pic_length
        else:
            min, max = center - 0.5 * cliped_length, center + 0.5 * cliped_length + 1
        return int(min), int(max)

    if boundary_width < cliped_size[0] and boundary_height < cliped_size[1]:
        left, right = get_boundary(center_x, cliped_size[0] - 1, pic_size[0])
        upper, lower = get_boundary(center_y, cliped_size[1] - 1, pic_size[1])
        apply_resize = False
    # 基于病灶中心按照病灶的边界的最长边进行切片(正方形), 随后resize为32x32
    else:
        max_length = max(boundary_height, boundary_width)
        left, right = get_boundary(center_x, max_length, pic_size[0])
        upper, lower = get_boundary(center_y, max_length, pic_size[1])
        apply_resize = True
    return left, upper, right, lower, apply_resize


def only_center_contour(mask: np.ndarray, center: Tuple[float, float]):
    """在图像中, 对具有多个分割区域的mask, 消除其余mask, 仅保留中心部分的mask."""

    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)

    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    if len(contours) != 1:
        draw_list = []
        for idx, contour in enumerate(contours):
            contour = np.squeeze(contour)
            if len(contour.shape) == 1:
                draw_list.append(idx)
                continue
            contour_right, contour_lower = np.max(contour, axis=0)
            contour_left, contour_upper = np.min(contour, axis=0)
            if (
                center[0] < contour_left
                or center[0] > contour_right
                or center[1] < contour_upper
                or center[1] > contour_lower
            ):
                draw_list.append(idx)
        for d in draw_list:
            cv2.drawContours(mask, contours, d, (0, 0, 0), cv2.FILLED)
    return mask


def get_resampled_SUVbw_from_PETCT(
    PET_files: List[str], CT_image: sitk.Image
) -> np.ndarray:
    """对于同一个患者的PETCT, PET将根据CT进行重采样到跟CT一样, 并计算PET的SUVbw.

    Args:
        PET_files (List[str]): 病患的一系列PET切片文件
        CT (sitk.Image): 病患的一系列CT, sitk.Image

    Returns:
        np.ndarray: 与CT一样的三维空间分辨率的suvbw
    """

    PET_image = read_serises_image(PET_files)
    PET_array = sitk.GetArrayFromImage(PET_image)

    # 计算每张PET slice的SUVbw
    SUVbw = np.zeros_like(PET_array, dtype=np.float32)
    for i in range(PET_array.shape[0]):
        SUVbw[i] = get_SUV_in_GE(PET_array[i], PET_files[i])[0]

    # 将suvbw变为图像, 并根据CT进行重采样.
    SUVbw_image = sitk.GetImageFromArray(SUVbw)
    SUVbw_image.CopyInformation(PET_image)
    resampled_SUVbw_image = resample_based_target_image(SUVbw_image, CT_image)

    return sitk.GetArrayFromImage(resampled_SUVbw_image)


""" series images的顺序是从肺部上面到下面, .nii.gz的顺序恰好相反, 从下面到上面."""

# 常量
SEGMENTATION_FILES = glob("D:/Desktop/Data/PET-CT/*/*.nii.gz")
LUNG_SLICE = pd.read_excel("D:/Desktop/Data/PET-CT/PET-CT.xlsx", "Sheet1")[
    ["肺部切片第一张", "肺部切片最后一张"]
].values

# 输出文件路径
data_folder = os.path.join(OUTPUT_FOLDER, "PETCT_20221001")
image_folder = os.path.join(OUTPUT_FOLDER, "PETCT_20221001", "images")
mkdirs([OUTPUT_FOLDER, data_folder, image_folder])
# reg 数据处理
for segmentation_file in SEGMENTATION_FILES:
    print("now start process file: ", segmentation_file)

    # 获取当前文件夹和上一级文件夹名
    segmentation_file_dir = os.path.dirname(segmentation_file)
    dir_name = os.path.basename(segmentation_file_dir)

    # 肺部切片
    slice_start, slice_end = LUNG_SLICE[int(dir_name) - 1]
    # 计算肺部切片长度
    slice_length = slice_end - slice_start + 1

    # 读取分割标签文件, 并翻转, 因为顺序与PETCT相反
    segmentation_array = sitk.GetArrayFromImage(sitk.ReadImage(segmentation_file))
    # 验证标签数据跟肺部切片是否匹配
    if slice_length != segmentation_array.shape[0]:
        print("slice length not match segmentation file!!! process next one.")
        break
    segmentation_array = np.flip(segmentation_array, axis=0)

    # 获取相应患者的CT图像
    series_CT_files = glob(os.path.join(segmentation_file_dir, "CT*"))

    # 读取CT图像
    CT_image = read_serises_image(series_CT_files)
    CT_array = sitk.GetArrayFromImage(CT_image)

    # 取出CT肺部切片, 文件名由000开始编号, 故如此切片
    lung_CT_files = series_CT_files[slice_start : slice_end + 1]
    lung_CT_array = CT_array[slice_start : slice_end + 1]

    # 计算肺部的HU
    lung_HU = lung_CT_array.astype(np.float32)

    # 获取相应患者的PET图像
    series_PET_files = glob(os.path.join(segmentation_file_dir, "PET*"))

    # 计算SUVbw
    SUVbw = get_resampled_SUVbw_from_PETCT(series_PET_files, CT_image)

    # 取出SUVbw肺部切片, 文件名由000开始编号, 故如此切片
    lung_SUVbw = SUVbw[slice_start : slice_end + 1]

    # 对每个肺部切片开始处理
    for i in range(slice_length):

        # 获取当前分割标签d
        current_segmaentation_array = segmentation_array[i]

        # 有分割图时，进行处理
        if 0 != np.max(current_segmaentation_array):

            # 获取当前CT文件名
            current_CT_filename = lung_CT_files[i].split("\\")[-1][:-4]
            print(
                "%s_%s lung slice file is processing!" % (dir_name, current_CT_filename)
            )

            # 获取当前的HU和SUVbw
            current_HU = lung_HU[i]
            current_HU_img = ct2image(copy.deepcopy(current_HU), 0, 2000, True)
            current_SUVbw = lung_SUVbw[i]

            # 由于每张图片可能存在多个病灶，所以需要定位出每个病灶并计算出每个病灶的suv max，min，mean
            contours, hierarchy = cv2.findContours(
                current_segmaentation_array.astype(np.uint8),
                cv2.RETR_LIST,
                cv2.CHAIN_APPROX_NONE,
            )

            # 将每个轮廓绘制在 CT_img 上
            cv2.drawContours(current_HU_img, contours, -1, (255, 255, 255), 1)
            cv2.imwrite(
                os.path.join(image_folder, f"{dir_name}_{current_CT_filename}.jpg"),
                current_HU_img,
                [int(cv2.IMWRITE_JPEG_QUALITY), 100],
            )
            cv2.imwrite(
                os.path.join(
                    image_folder, f"{dir_name}_{current_CT_filename}_mask.jpg"
                ),
                current_segmaentation_array * 255,
                [int(cv2.IMWRITE_JPEG_QUALITY), 100],
            )

            # 处理每个病灶
            for idx, contour in enumerate(contours):

                contour = np.squeeze(contour)
                if len(contour.shape) == 1:
                    break

                # 获得每个病灶的大致位置
                contour_right, contour_lower = np.max(contour, axis=0)
                contour_left, contour_upper = np.min(contour, axis=0)

                # 在mask上仅保留一个轮廓
                only_one_segmentation = only_center_contour(
                    current_segmaentation_array,
                    [
                        (contour_left + contour_right) * 0.5,
                        (contour_upper + contour_lower) * 0.5,
                    ],
                )

                # 根据mask计算最大直径
                pulmonary_nodules = load_json("pulmonary_nodules.json")
                spacing = pulmonary_nodules[dir_name]["Spacing"]
                max_distance = 0
                points = []
                for i in range(len(contour)):
                    for j in range(i + 1, len(contour)):
                        d = np.abs(contour[i] - contour[j]) * spacing[:2]
                        md = sqrt(d[0] ** 2 + d[1] ** 2)
                        if md > max_distance:
                            max_distance = md
                            points = [contour[i].tolist(), contour[j].tolist()]

                # 保存仅一个轮廓的 mask 图像文件
                only_one_segmentation_image = only_one_segmentation * 255
                cv2.imwrite(
                    os.path.join(
                        image_folder,
                        f"{dir_name}_{current_CT_filename}_{str(idx).zfill(2)}_mask.jpg",
                    ),
                    only_one_segmentation_image,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 100],
                )
                cv2.line(
                    only_one_segmentation_image,
                    points[0],
                    points[1],
                    color=(127, 127, 127),
                    thickness=1,
                )
                cv2.imwrite(
                    os.path.join(
                        image_folder,
                        f"{dir_name}_{current_CT_filename}_{str(idx).zfill(2)}_mask_.jpg",
                    ),
                    only_one_segmentation_image,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 100],
                )

                # 根据仅有一个轮廓的mask, 计算每个病灶区域的 suv max, suv mean, suv min
                masked_SUVbw = np.ma.masked_where(
                    only_one_segmentation == 0, current_SUVbw
                )
                SUVbw_max = np.max(masked_SUVbw)
                SUVbw_min = np.min(masked_SUVbw)
                SUVbw_mean = np.mean(masked_SUVbw)

                pulmonary_nodules[dir_name][
                    current_CT_filename + "_" + str(idx).zfill(2)
                ] = {
                    "distance": max_distance,
                    "points": points,
                    "suv": [
                        SUVbw_max.tolist(),
                        SUVbw_mean.tolist(),
                        SUVbw_min.tolist(),
                    ],
                }

                save_json("pulmonary_nodules.json", pulmonary_nodules)

                # 在CT和切割标签图中切出每个病灶
                left, upper, right, lower, apply_resize = crop_based_lesions(
                    [contour_left, contour_upper, contour_right, contour_lower],
                    cliped_size=(32, 32),
                )
                cropped_HU = current_HU[upper:lower, left:right]
                cropped_segmentation = only_one_segmentation[upper:lower, left:right]
                if apply_resize:
                    cropped_HU = cv2.resize(cropped_HU, (32, 32))
                    cropped_segmentation = cv2.resize(cropped_segmentation, (32, 32))
                    print("there is one need to resize to 32x32!!")

                # 保存npz文件: cropped HU(32x32), cropped segmentation(32x32), SUVbw max, SUVbw mean, SUVbw min
                np.savez(
                    os.path.join(
                        data_folder,
                        f"{dir_name}_{current_CT_filename}_{str(idx).zfill(2)}",
                    ),
                    hounsfield_unit=cropped_HU,
                    mask=cropped_segmentation,
                    SUVmax=SUVbw_max,
                    SUVmean=SUVbw_mean,
                    SUVmin=SUVbw_min,
                )

                # 保存图像文件
                cv2.imwrite(
                    os.path.join(
                        image_folder,
                        f"{dir_name}_{current_CT_filename}_{str(idx).zfill(2)}_croped.jpg",
                    ),
                    ct2image(copy.deepcopy(cropped_HU), 0, 2000, True),
                    [int(cv2.IMWRITE_JPEG_QUALITY), 100],
                )
                cv2.imwrite(
                    os.path.join(
                        image_folder,
                        f"{dir_name}_{current_CT_filename}_{str(idx).zfill(2)}_croped_mask.jpg",
                    ),
                    cropped_segmentation * 255,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 100],
                )
