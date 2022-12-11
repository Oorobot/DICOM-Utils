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
    resample_based_target_image,
    resample_based_spacing,
    resameple_based_size,
)
from utils.utils import delete, load_json, mkdir, rename, save_json, to_pinyin

normal_folder_name = "D:/admin/Desktop/Data/PETCT-FRI/NormalData"
processed_folder_name = "D:/admin/Desktop/Data/PETCT-FRI/ProcessedData"
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
    z = body_mask.shape[0] + 1
    changed = False
    for i, slice in enumerate(body_mask):
        contours, _ = cv2.findContours(
            slice.astype(np.uint8),
            cv2.RETR_LIST,
            cv2.CHAIN_APPROX_NONE,
        )
        if len(contours) == 0 and changed:
            z = i
            changed = True
        for contour in contours:
            contour = np.squeeze(contour, axis=1)
            x1, y1 = np.min(contour, axis=0)
            x2, y2 = np.max(contour, axis=0)
            x1_ = x1 if x1 < x1_ else x1_
            y1_ = y1 if y1 < y1_ else y1_
            x2_ = x2 if x2 > x2_ else x2_
            y2_ = y2 if y2 > y2_ else y2_
    return x1_, y1_ + 1, x2_, y2_ + 1, 0, z


""" 批量计算 body_mask """
# sitk.ProcessObject_SetGlobalDefaultNumberOfThreads(8)
# folders = os.listdir(normal_folder_name)
# for folder in folders:
#     if int(folder) < 820:
#         continue
#     print(f"=> 开始处理 {folder}")
#     ct_path = os.path.join(process_folder_name, f"{folder}_CT.nii.gz")
#     suv_path = os.path.join(process_folder_name, f"{folder}_SUVbw.nii.gz")

#     ct_image = sitk.ReadImage(ct_path)
#     suv_image = sitk.ReadImage(suv_path)

#     print("==> 进行重采样")
#     # 将SUV图像重采样到与CT相同大小
#     resampled_suv_image = resample(suv_image, ct_image)
#     body_mask = get_body(ct_image, resampled_suv_image)
#     print("==> 写入 body mask closing max")
#     sitk.WriteImage(body_mask, f"./Files/body_mask2/{folder}_body_mask.nii.gz")


""" 获取三维定位框的标注 """
# object_labels = glob(os.path.join(processed_folder_name, "*Label_Object.nii.gz"))
# body_labels = glob(os.path.join(processed_folder_name, "*Label_Body.nii.gz"))

# label_info: dict = load_json("label_info.json")
# for no, info in label_info.items():
#     if no == "category":
#         continue
#     x1, y1, x2, y2, z1, z2 = info["crop_position"]
#     info["crop_position"] = [x1, y1, z1, x2, y2, z2]
# save_json("label_info_.json", label_info)
# annotations = {}
# for new_label in new_labels:
#     no = new_label.split("_")[0]
#     new_label_image = sitk.ReadImage(new_label)
#     classes, annotations = get_3D_annotation(new_label)
DATA_FOLDER = "D:/admin/Desktop/Data/PETCT-FRI/ProcessedData"
LABEL_INFOMATION = load_json("label_info.json")


def data_propress(image_id, input_shape=[128, 196, 196]):
    crop_position = LABEL_INFOMATION[image_id]["crop_position"]
    # ------------------------------#
    #   读取图像
    # ------------------------------#
    ct = sitk.ReadImage(os.path.join(DATA_FOLDER, f"{image_id}_CT.nii.gz"))
    pet = sitk.ReadImage(os.path.join(DATA_FOLDER, f"{image_id}_SUVbw.nii.gz"))
    label_body = sitk.ReadImage(
        os.path.join(DATA_FOLDER, f"{image_id}_Label_Body.nii.gz")
    )
    # 根据 crop_position 进行裁剪 ct
    crop_index = (np.array(crop_position[0:3]) + 1).tolist()
    crop_size = (
        np.array(crop_position[3:6]) - np.array(crop_position[0:3]) - 2
    ).tolist()

    cropped_ct = sitk.Extract(ct, size=crop_size, index=crop_index)
    # 重采样PET
    resampled_pet = resample_based_target_image(pet, ct)
    # 根据 crop_position 进行裁剪 pet
    cropped_pet = sitk.Extract(resampled_pet, size=crop_size, index=crop_index)
    # 根据 crop_position 进行裁剪 pet
    cropped_body = sitk.Extract(label_body, size=crop_size, index=crop_index)

    print(cropped_ct.GetSize())
    # 计算新的 resample 大小 crop_size = [W, H, D]
    iw, ih, id = (
        crop_position[3] - crop_position[0],
        crop_position[4] - crop_position[1],
        crop_position[5] - crop_position[2],
    )
    d, h, w = input_shape

    scale = min(w / iw, h / ih, d / id)
    nw = int(iw * scale)
    nh = int(ih * scale)
    nd = int(id * scale)

    # resample 到 新的大小去
    resampled_ct = resameple_based_size(cropped_ct, [nw, nh, nd])
    resampled_pet = resameple_based_size(cropped_pet, [nw, nh, nd])
    resampled_body = resameple_based_size(cropped_body, [nw, nh, nd], True)
    print(resampled_ct.GetSize())
    dx = (w - nw) // 2
    dx_ = w - nw - dx
    dy = (h - nh) // 2
    dy_ = h - nh - dy
    dz = (d - nd) // 2
    dz_ = d - nd - dz
    # 提取array, 进行padding, 随后进行预处理
    resampled_ct_array = sitk.GetArrayFromImage(resampled_ct)
    resampled_pet_array = sitk.GetArrayFromImage(resampled_pet)
    resampled_body_array = sitk.GetArrayFromImage(resampled_body)
    resampled_ct_array = (
        resampled_ct_array * (resampled_body_array) + (1 - resampled_body_array) * -1000
    )
    resampled_pet_array = resampled_pet_array * resampled_body_array

    ct_array = np.pad(
        resampled_ct_array, ((dz, dz_), (dy, dy_), (dx, dx_)), constant_values=-1000
    )
    pet_array = np.pad(resampled_pet_array, ((dz, dz_), (dy, dy_), (dx, dx_)))

    ct_image = sitk.GetImageFromArray(ct_array)
    ct_image.SetOrigin(resampled_ct.GetOrigin())
    ct_image.SetSpacing(resampled_ct.GetSpacing())
    ct_image.SetDirection(resampled_ct.GetDirection())
    pet_image = sitk.GetImageFromArray(pet_array)
    pet_image.SetOrigin(resampled_pet.GetOrigin())
    pet_image.SetSpacing(resampled_pet.GetSpacing())
    pet_image.SetDirection(resampled_pet.GetDirection())
    sitk.WriteImage(ct_image, f"{image_id}.nii.gz")
    sitk.WriteImage(pet_image, f"{image_id}_.nii.gz")


data_propress("001")
