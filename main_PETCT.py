import os
from glob import glob
from typing import List

import cv2
import numpy as np
import pandas as pd
import SimpleITK as sitk

from dicom import get_SUVbw_in_GE, read_serises_image
from utils import crop_based_lesions, only_center_contour, save_images


def resample(
    img: sitk.Image, tar_img: sitk.Image, is_label: bool = False
) -> sitk.Image:

    resamlper = sitk.ResampleImageFilter()
    resamlper.SetReferenceImage(tar_img)
    resamlper.SetOutputPixelType(sitk.sitkFloat32)
    if is_label:
        resamlper.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        # resamlper.SetInterpolator(sitk.sitkBSpline)
        resamlper.SetInterpolator(sitk.sitkLinear)
    return resamlper.Execute(img)


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
        SUVbw[i] = get_SUVbw_in_GE(PET_array[i], PET_files[i])

    # 将suvbw变为图像, 并根据CT进行重采样.
    SUVbw_image = sitk.GetImageFromArray(SUVbw)
    SUVbw_image.SetOrigin(PET_image.GetOrigin())
    SUVbw_image.SetSpacing(PET_image.GetSpacing())
    SUVbw_image.SetDirection(PET_image.GetDirection())
    resampled_SUVbw_image = resample(SUVbw_image, CT_image)

    return sitk.GetArrayFromImage(resampled_SUVbw_image)


""" series images的顺序是从肺部上面到下面, .nii.gz的顺序恰好相反, 从下面到上面."""

# 常量
SEGMENTATION_FILES = glob("PET-CT/*/*.nii.gz")
LUNG_SLICE = pd.read_excel("PET-CT/PET-CT.xlsx", "Sheet1")[
    ["肺部切片第一张", "肺部切片最后一张"]
].values

# reg 数据处理
for segmentation_file in SEGMENTATION_FILES:
    print("now start process file: ", segmentation_file)

    # 获取当前文件夹和上一级文件夹名
    segmentation_file_dir = os.path.dirname(segmentation_file)
    dir_name = segmentation_file_dir.split("\\")[-1]

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
            current_SUVbw = lung_SUVbw[i]

            # 获取masked后的CT图像
            masked_CT_image_1 = np.ma.masked_where(
                current_segmaentation_array == 0, current_HU
            )
            masked_CT_image_2 = np.ma.masked_where(
                current_segmaentation_array == 1, current_HU
            )

            # 保存masked后的CT图像
            save_images(
                [masked_CT_image_1, masked_CT_image_2],
                ["masked CT image 1", "masked CT image 2"],
                ["bone", "bone"],
                "ProcessedData/image/%s_%s_mask.png" % (dir_name, current_CT_filename),
            )

            # 由于每张图片可能存在多个病灶，所以需要定位出每个病灶并计算出每个病灶的suv max，min，mean
            contours, hierarchy = cv2.findContours(
                current_segmaentation_array.astype(np.uint8),
                cv2.RETR_LIST,
                cv2.CHAIN_APPROX_NONE,
            )

            # 处理每个病灶
            for idx, contour in enumerate(contours):
                contour = np.squeeze(contour)
                if len(contour.shape) == 1:
                    break
                contour_right, contour_lower = np.max(contour, axis=0)
                contour_left, contour_upper = np.min(contour, axis=0)

                # 计算每个病灶区域的 suv max, suv mean, suv min
                masked_SUVbw = np.ma.masked_where(
                    current_segmaentation_array == 0, current_SUVbw
                )
                cropped_masked_SUVbw = masked_SUVbw[
                    contour_upper : contour_lower + 1, contour_left : contour_right + 1
                ]
                SUVbw_max = np.max(cropped_masked_SUVbw)
                SUVbw_min = np.min(cropped_masked_SUVbw)
                SUVbw_mean = np.mean(cropped_masked_SUVbw)

                # 在CT和切割标签图中切出每个病灶
                left, upper, right, lower, apply_resize = crop_based_lesions(
                    [contour_left, contour_upper, contour_right, contour_lower],
                    cliped_size=(32, 32),
                )
                cropped_HU = current_HU[upper:lower, left:right]
                print(cropped_HU.shape)
                cropped_segmentation = current_segmaentation_array[
                    upper:lower, left:right
                ]
                if apply_resize:
                    cropped_HU = cv2.resize(cropped_HU, (32, 32))
                    cropped_segmentation = cv2.resize(cropped_segmentation, (32, 32))
                    print("there is one need to resize to 32x32!!")

                # seg 仅保留一个中心的病灶
                cropped_segmentation_only_one = only_center_contour(
                    cropped_segmentation, (15.5, 15.5)
                )

                # 保存图像文件
                save_images(
                    [cropped_HU, cropped_segmentation, cropped_segmentation_only_one],
                    ["img", "seg", "seg_one"],
                    ["bone", "gray", "gray"],
                    "_ProcessedData_/image/%s_%s_%s_cliped.png"
                    % (dir_name, current_CT_filename, str(idx).zfill(2)),
                )

                # 保存npz文件: cropped HU(32x32), cropped segmentation(32x32), SUVbw max, SUVbw mean, SUVbw min
                np.savez(
                    "_ProcessedData_/regression/%s_%s_%s.npz"
                    % (dir_name, current_CT_filename, str(idx).zfill(2),),
                    HU=cropped_HU,
                    segmentation=cropped_segmentation_only_one,
                    max=SUVbw_max,
                    mean=SUVbw_mean,
                    min=SUVbw_min,
                )
