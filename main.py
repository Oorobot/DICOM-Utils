import copy
import datetime
import math
import os
import shutil
import zipfile
from glob import glob

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydicom
import SimpleITK as sitk
from mlxtend.evaluate import mcnemar, mcnemar_table  # 用于计算显著性水平 p
from skimage import measure

from utils.dicom import (
    get_3D_annotation,
    get_patient_info,
    get_pixel_array,
    get_pixel_value,
    get_SUVbw_in_GE,
    read_serises_image,
    resample,
    resample_spacing,
)
from utils.utils import delete, load_json, mkdir, rename, save_json, to_pinyin

folder_name = "D:/admin/Desktop/Data/PETCT-FRI/NormalData"
fri_xlsx = "D:/admin/Desktop/Data/PETCT-FRI/PET-FRI.xlsx"


def get_max_component(mask_image: sitk.Image) -> sitk.Image:
    # 得到mask中的多个连通量
    cc_filter = sitk.ConnectedComponentImageFilter()
    cc_filter.SetNumberOfThreads(8)
    cc_filter.SetFullyConnected(True)

    output_image = cc_filter.Execute(mask_image)

    # 计算不同连通图的大小
    lss_filter = sitk.LabelShapeStatisticsImageFilter()
    lss_filter.SetNumberOfThreads(8)
    lss_filter.Execute(output_image)

    label_num = cc_filter.GetObjectCount()
    max_label = 0
    max_num = 0

    for i in range(1, label_num + 1):
        num = lss_filter.GetNumberOfPixels(i)
        if num > max_num:
            max_label = i
            max_num = num
    output_array = sitk.GetArrayFromImage(output_image)
    max_component = (output_array == max_label).astype(np.uint8)

    max_component_image = sitk.GetImageFromArray(max_component)
    max_component_image.CopyInformation(mask_image)

    return max_component_image


def get_binary_image(ct_image: sitk.Image, threshold: int = -200) -> sitk.Image:
    """
    CT threshold = -200, SUVbw threshold = 1e-2
    """
    ct_array = sitk.GetArrayFromImage(ct_image)
    binary_ct_array = (ct_array > threshold).astype(np.uint8)
    binary_ct_image = sitk.GetImageFromArray(binary_ct_array)
    binary_ct_image.CopyInformation(ct_image)
    return binary_ct_image


def get_binary_morphological_closing(mask_image: sitk.Image):
    bmc_filter = sitk.BinaryMorphologicalClosingImageFilter()
    bmc_filter.SetKernelType(sitk.sitkBall)
    bmc_filter.SetKernelRadius(3)
    bmc_filter.SetForegroundValue(1)
    bmc_filter.SetNumberOfThreads(8)
    return bmc_filter.Execute(mask_image)


def get_binary_morphological_opening(mask_image: sitk.Image):
    bmo_filter = sitk.BinaryMorphologicalOpeningImageFilter()
    bmo_filter.SetKernelType(sitk.sitkBall)
    bmo_filter.SetKernelRadius(3)
    bmo_filter.SetForegroundValue(1)
    bmo_filter.SetNumberOfThreads(8)
    return bmo_filter.Execute(mask_image)


def get_body(
    ct_image: sitk.Image,
    suv_image: sitk.Image,
):
    ct_binary = get_binary_image(ct_image, -200)
    suv_binary = get_binary_image(suv_image, 3e-2)

    # 对CT进行闭操作，取最大连通量
    ct_binary_closing_max = get_max_component(
        get_binary_morphological_closing(ct_binary)
    )
    # 对SUV进行闭操作，取最大连通量
    suv_binary_closing_max = get_max_component(
        get_binary_morphological_closing(suv_binary)
    )

    # total mask = ct mask ∪ suv mask
    total_mask = sitk.Or(ct_binary_closing_max, suv_binary_closing_max)
    # machine mask = ct mask - suv mask
    machine_mask = sitk.And(total_mask, sitk.Not(suv_binary_closing_max))

    # 对 machine mask 进行闭包
    machine_mask_closing = get_binary_morphological_closing(machine_mask)

    # body mask = total mask - machine mask
    body_mask = sitk.And(total_mask, sitk.Not(machine_mask_closing))
    # 分别进行闭操作和开操作，然后进行取最大连通量
    body_mask_closing = get_max_component(get_binary_morphological_closing(body_mask))
    # body_mask_opening = get_max_component(get_binary_morphological_opening(body_mask))

    # sitk.WriteImage(total_mask, "./Files/total_mask.nii.gz")
    # sitk.WriteImage(machine_mask, "./Files/machine_mask.nii.gz")
    # sitk.WriteImage(machine_mask_closing, "./Files/machine_mask_closing.nii.gz")
    # sitk.WriteImage(body_mask, "./Files/body_mask.nii.gz")
    # sitk.WriteImage(ct_binary, "./Files/ct_binary.nii.gz")
    # sitk.WriteImage(ct_binary_closing_max, "./Files/ct_binary_closing_max.nii.gz")
    # sitk.WriteImage(suv_binary, "./Files/suv_binary.nii.gz")
    # sitk.WriteImage(suv_binary_closing_max, "./Files/suv_binary_closing_max.nii.gz")

    # sitk.WriteImage(body_mask_closing, "./Files/body_mask_closing.nii.gz")
    # sitk.WriteImage(body_mask_opening, "./Files/body_mask_opening.nii.gz")
    return body_mask_closing


if __name__ == "__main__":
    folders = os.listdir(folder_name)
    for folder in folders:
        print(f"=> 开始处理 {folder}")
        ct_path = os.path.join(folder_name, folder, f"{folder}_CT.nii.gz")
        suv_path = os.path.join(folder_name, folder, f"{folder}_SUVbw.nii.gz")

        ct_image = sitk.ReadImage(ct_path)
        suv_image = sitk.ReadImage(suv_path)

        print("==> 进行重采样")
        # 将SUV图像重采样到与CT相同大小
        resampled_suv_image = resample(suv_image, ct_image)
        print("==> 计算 body mask closing max")
        body_mask = get_body(ct_image, resampled_suv_image)
        print("==> 写入 body mask closing max")
        sitk.WriteImage(
            body_mask, f"./Files/body_mask/{folder}_body_closing_max.nii.gz"
        )
