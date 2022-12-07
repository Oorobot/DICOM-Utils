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
    cc_filter.SetFullyConnected(True)

    output_image = cc_filter.Execute(mask_image)

    # 计算不同连通图的大小
    lss_filter = sitk.LabelShapeStatisticsImageFilter()
    lss_filter.Execute(output_image)

    label_num = cc_filter.GetObjectCount()
    max_label = 0
    max_num = 0

    for i in range(1, label_num + 1):
        num = lss_filter.GetNumberOfPixels(i)
        if num > max_num:
            max_label = i
            max_num = num
    return sitk.Equal(output_image, max_label)


def get_binary_image(image: sitk.Image, threshold: int = -200) -> sitk.Image:
    """
    CT threshold = -200, SUVbw threshold = 1e-2
    """
    return sitk.BinaryThreshold(image, lowerThreshold=threshold, upperThreshold=1e8)


def get_binary_morphological_closing(mask_image: sitk.Image, kernel_radius=2):
    bmc_filter = sitk.BinaryMorphologicalClosingImageFilter()
    bmc_filter.SetKernelType(sitk.sitkBall)
    bmc_filter.SetKernelRadius(kernel_radius)
    bmc_filter.SetForegroundValue(1)
    return bmc_filter.Execute(mask_image)


def get_binary_morphological_opening(mask_image: sitk.Image, kernel_radius=2):
    bmo_filter = sitk.BinaryMorphologicalOpeningImageFilter()
    bmo_filter.SetKernelType(sitk.sitkBall)
    bmo_filter.SetKernelRadius(kernel_radius)
    bmo_filter.SetForegroundValue(1)
    return bmo_filter.Execute(mask_image)


def get_body(
    ct_image: sitk.Image,
    suv_image: sitk.Image,
):
    # 忽略其中进行闭操作和最大连通量
    # CT(去除机床) = CT binary ∩ SUV binary; CT(去除伪影) = CT binary(去除机床) ∪ SUV binary;
    # CT(去除伪影) ≈ SUV binary
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

    ct_no_machine = sitk.And(ct_binary_closing_max, suv_binary_closing_max)
    # 取最大连通量
    ct_no_machine_max = get_max_component(ct_no_machine)

    # 使用超大半径的闭操作，消除伪影
    ct_no_machine_max_closing = get_binary_morphological_closing(ct_no_machine_max, 20)
    return ct_no_machine_max_closing


def preprosses_based_body_mask(body_mask: np.ndarray):
    y1_, x1_ = body_mask.shape[1:]
    y2_, x2_ = -1, -1
    for slice in body_mask:
        contours = cv2.findContours(
            slice.astype(np.uint8),
            cv2.RETR_LIST,
            cv2.CHAIN_APPROX_NONE,
        )
        for contour in contours:
            contour = np.squeeze(contour, axis=1)
            x1, y1 = np.min(contour, axis=0)
            x2, y2 = np.max(contour, axis=0)
            x1_ = x1 if x1 < x1_ else x1_
            y1_ = y1 if y1 < y1_ else y1_
            x2_ = x2 if x2 < x2_ else x2_
            y2_ = y2 if y2 < y2_ else y2_
    return x1_, y1_, x2_, y2_


sitk.ProcessObject_SetGlobalDefaultNumberOfThreads(8)

""" 批量计算 body_mask   """
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
    sitk.WriteImage(body_mask, f"./Files/body_mask2/{folder}_body_mask.nii.gz")
