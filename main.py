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
    read_serises_image,
    resample_based_target_image,
    resample_based_spacing,
    resameple_based_size,
)
from utils.utils import delete, load_json, mkdir, rename, save_json, to_pinyin

normal_folder_name = "D:/admin/Desktop/Data/PETCT-FRI/NormalData"
processed_folder_name = "D:/admin/Desktop/Data/PETCT-FRI/ProcessedData"
fri_xlsx = "D:/admin/Desktop/Data/PETCT-FRI/PET-FRI.xlsx"


# def data_propress(image_id, input_shape=[128, 196, 196]):
#     crop_position = LABEL_INFOMATION[image_id]["crop_position"]
#     # ------------------------------#
#     #   读取图像
#     # ------------------------------#
#     ct = sitk.ReadImage(os.path.join(DATA_FOLDER, f"{image_id}_CT.nii.gz"))
#     pet = sitk.ReadImage(os.path.join(DATA_FOLDER, f"{image_id}_SUVbw.nii.gz"))
#     label_body = sitk.ReadImage(
#         os.path.join(DATA_FOLDER, f"{image_id}_Label_Body.nii.gz")
#     )
#     # 根据 crop_position 进行裁剪 ct
#     crop_index = (np.array(crop_position[0:3]) + 1).tolist()
#     crop_size = (
#         np.array(crop_position[3:6]) - np.array(crop_position[0:3]) - 2
#     ).tolist()

#     cropped_ct = sitk.Extract(ct, size=crop_size, index=crop_index)
#     # 重采样PET
#     resampled_pet = resample_based_target_image(pet, ct)
#     # 根据 crop_position 进行裁剪 pet
#     cropped_pet = sitk.Extract(resampled_pet, size=crop_size, index=crop_index)
#     # 根据 crop_position 进行裁剪 pet
#     cropped_body = sitk.Extract(label_body, size=crop_size, index=crop_index)

#     print(cropped_ct.GetSize())
#     # 计算新的 resample 大小 crop_size = [W, H, D]
#     iw, ih, id = (
#         crop_position[3] - crop_position[0],
#         crop_position[4] - crop_position[1],
#         crop_position[5] - crop_position[2],
#     )
#     d, h, w = input_shape

#     scale = min(w / iw, h / ih, d / id)
#     nw = int(iw * scale)
#     nh = int(ih * scale)
#     nd = int(id * scale)

#     # resample 到 新的大小去
#     resampled_ct = resameple_based_size(cropped_ct, [nw, nh, nd])
#     resampled_pet = resameple_based_size(cropped_pet, [nw, nh, nd])
#     resampled_body = resameple_based_size(cropped_body, [nw, nh, nd], True)
#     print(resampled_ct.GetSize())
#     dx = (w - nw) // 2
#     dx_ = w - nw - dx
#     dy = (h - nh) // 2
#     dy_ = h - nh - dy
#     dz = (d - nd) // 2
#     dz_ = d - nd - dz
#     # 提取array, 进行padding, 随后进行预处理
#     resampled_ct_array = sitk.GetArrayFromImage(resampled_ct)
#     resampled_pet_array = sitk.GetArrayFromImage(resampled_pet)
#     resampled_body_array = sitk.GetArrayFromImage(resampled_body)
#     resampled_ct_array = (
#         resampled_ct_array * (resampled_body_array) + (1 - resampled_body_array) * -1000
#     )
#     resampled_pet_array = resampled_pet_array * resampled_body_array

#     ct_array = np.pad(
#         resampled_ct_array, ((dz, dz_), (dy, dy_), (dx, dx_)), constant_values=-1000
#     )
#     pet_array = np.pad(resampled_pet_array, ((dz, dz_), (dy, dy_), (dx, dx_)))

#     ct_image = sitk.GetImageFromArray(ct_array)
#     ct_image.SetOrigin(resampled_ct.GetOrigin())
#     ct_image.SetSpacing(resampled_ct.GetSpacing())
#     ct_image.SetDirection(resampled_ct.GetDirection())
#     pet_image = sitk.GetImageFromArray(pet_array)
#     pet_image.SetOrigin(resampled_pet.GetOrigin())
#     pet_image.SetSpacing(resampled_pet.GetSpacing())
#     pet_image.SetDirection(resampled_pet.GetDirection())
#     sitk.WriteImage(ct_image, f"{image_id}.nii.gz")
#     sitk.WriteImage(pet_image, f"{image_id}_.nii.gz")

# 将label的数据类型转换为 int32

labels = glob(os.path.join(processed_folder_name, "*Label*"))
for label in labels:
    label_image = sitk.ReadImage(label)
    resampeld_label = resample_based_target_image(label_image, label_image, True)
    sitk.WriteImage(resampeld_label, os.path.join("./Files", os.path.basename(label)))
