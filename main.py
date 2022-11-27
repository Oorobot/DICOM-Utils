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
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
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


"""将label重采样到CT大小"""
# no = "098"
# print("正在处理" + no)
# ct = os.path.join(folder_name, no, f"{no}_CT.nii.gz")
# ct_image = sitk.ReadImage(ct)
# label = sitk.ReadImage(os.path.join(folder_name, no, "Untitled.nii.gz"))
# resample_label = resample(label, ct_image, True)
# sitk.WriteImage(resample_label, os.path.join(folder_name, no, f"{no}_CT_Label.nii.gz"))

"""数据处理"""
# for dir in os.listdir(folder_name):
#     # 读取PET数据，进行重采样，得到SUVbw样本
#     if not os.path.exists(os.path.join(folder_name, dir, "SUVbw.nii.gz")):
#         print(f"==> {dir} - start to generate SUVbw.nii.gz")
#         pet_files = glob(os.path.join(folder_name, dir, "PET", "*"))
#         pet_image = read_serises_image(pet_files)
#         pet_array = sitk.GetArrayFromImage(pet_image)
#         suvbw_array = np.zeros_like(pet_array, dtype=np.float32)
#         for i in range(len(pet_array)):
#             suvbw_array[i] = get_SUVbw_in_GE(pet_array[i], pet_files[i])
#         suvbw_image = sitk.GetImageFromArray(suvbw_array)
#         if np.min(suvbw_array) < 0:
#             print("suvbw 存在问题：", os.path.join(folder_name, dir))
#         suvbw_image.SetOrigin(pet_image.GetOrigin())
#         suvbw_image.SetSpacing(pet_image.GetSpacing())
#         suvbw_image.SetDirection(pet_image.GetDirection())
#         resampled_suvbw_image = resample(suvbw_image, pet_image)
#         sitk.WriteImage(
#             resampled_suvbw_image, os.path.join(folder_name, dir, "SUVbw.nii.gz")
#         )
#     # 将CT的DICOM转换为nii.gz
#     if not os.path.exists(os.path.join(folder_name, dir, "CT.nii.gz")):
#         print(f"==> {dir} - start to generate CT.nii.gz")
#         ct_files = glob(os.path.join(folder_name, dir, "CT", "*"))
#         ct_image = read_serises_image(ct_files)
#         sitk.WriteImage(ct_image, os.path.join(folder_name, dir, "CT.nii.gz"))

"""记录标注数据"""
# class_name = {1: "fraction", 2: "bladder", 3: "other", 4: "4", 5: "5", 6: "6"}
# labels = glob(os.path.join(folder_name, "*", "*Label.nii.gz"))
# annotations = load_json("./Files/resampled_FRI/annotations.json")
# output_folder = "./Files/resampled_FRI"
# mkdir(output_folder)

# for label in labels:
#     folder = os.path.dirname(label)
#     no = os.path.basename(folder)
#     if int(no) < 0:
#         continue
#     ct = os.path.join(folder, f"{no}_CT.nii.gz")
#     pet = os.path.join(folder, f"{no}_SUVbw.nii.gz")
#     resampled_label = os.path.join(output_folder, f"{no}_rLabel.nii.gz")
#     if os.path.exists(resampled_label):
#         resampled_label = sitk.ReadImage(resampled_label)
#     else:
#         # 读取CT数据
#         ct_image = sitk.ReadImage(ct)
#         suvbw_image = sitk.ReadImage(pet)
#         label_image = sitk.ReadImage(label)
#         # 进行重采样到均为 spacing  1x1x1 cm
#         resampled_ct = resample_spacing(ct_image)
#         resampled_suvbw = resample(suvbw_image, resampled_ct)
#         resampled_label = resample(label_image, resampled_ct, True)
#         # 写入重采样的文件
#         if not os.path.exists(os.path.join(output_folder, f"{no}_rCT.nii.gz")):
#             sitk.WriteImage(
#                 resampled_ct, os.path.join(output_folder, f"{no}_rCT.nii.gz")
#             )
#         if not os.path.exists(os.path.join(output_folder, f"{no}_rSUVbw.nii.gz")):
#             sitk.WriteImage(
#                 resampled_suvbw, os.path.join(output_folder, f"{no}_rSUVbw.nii.gz")
#             )
#         sitk.WriteImage(
#             resampled_label, os.path.join(output_folder, f"{no}_rLabel.nii.gz")
#         )

#     print(f"正在处理{no}的3D标注数据...")
#     resample_label_array = sitk.GetArrayFromImage(resampled_label)
#     classes, locations = get_3D_annotation(resample_label_array)
#     for c, l in zip(classes, locations):
#         annotations[no] = {
#             "shape": resample_label_array.shape,
#             "annotations": [
#                 {"class": class_name[c], "location": l}
#                 for c, l in zip(classes, locations)
#             ],
#         }
#     save_json(os.path.join(output_folder, "annotations.json"), annotations)


"""剩余需要修订或标注的数据"""
# collected = []
# folders = os.listdir(folder_name)
# for folder in folders:
#     niigzs = glob(os.path.join(folder_name, folder, "*.nii.gz"))
#     if len(niigzs) == 2:
#         # print(folder)
#         collected.append(folder)
#     label_path = os.path.join(folder_name, folder, f"{folder}_CT_Label.nii.gz")
#     try:
#         label = get_pixel_value(label_path)
#         if np.max(label) > 2:
#             print("folder: ", folder)
#     except:
#         pass
# print("num: ", len(collected))

"""修改标签"""
folders = os.listdir(folder_name)
labels = glob(os.path.join(folder_name, "*", "*Label.nii.gz"))
patient_info = pd.read_excel(fri_xlsx, sheet_name="FRI-unique")

three2one = ["125"]
three2two = ["074", "102", "194", "345", "402", "414", "447", "615"]
three2four = ["024", "037", "045", "082", "089", "014", "108", "597"]
four2one = ["109"]
for label in labels:
    print(f"处理 {label} ……")
    no = os.path.basename(os.path.dirname(label))
    int_no = int(no)
    # 根据编号查询感染情况
    is_infected = np.squeeze(
        patient_info.query(f"No==@int_no")[["Final_diagnosis"]].values
    )

    label_image = sitk.ReadImage(label)
    label_array = sitk.GetArrayFromImage(label_image)

    # 深层复制 array, 用于修改标签, 以避免来回修改
    copy_array = copy.deepcopy(label_array)
    # 进行转换
    if no in three2one:
        label_array[copy_array == 3] = 1
    if no in three2two:
        label_array[copy_array == 3] = 2
    if no in three2four:
        label_array[copy_array == 3] = 4
    if no in four2one:
        label_array[copy_array == 4] = 1

    # 将 bladder 转换 2 -> 3
    label_array[copy_array == 2] = 3

    # 将病灶区域转换为 感染区域 1 和 非感染区域 2
    if is_infected == "F":
        label_array[copy_array == 1] = 2

    # array 转换回 image
    changed_image = sitk.GetImageFromArray(label_array)

    changed_image.CopyInformation(label_image)
    # print("===> label image: \n", label_image, "===> changed image: \n", changed_image)
    sitk.WriteImage(
        changed_image, os.path.join(folder_name, no, f"{no}_CT_Label_new.nii.gz")
    )


# def intersect_box(b1: np.ndarray, b2: np.ndarray, based_b1_correct_box: bool = True):
#     """
#     b1: (1, 6), 一个边界框
#     b2: (N, 7), 一堆边界框
#     """
#     N = b2.shape[0]
#     b1_x1, b1_y1, b1_z1, b1_x2, b1_y2, b1_z2 = (
#         b1[:, 0],
#         b1[:, 1],
#         b1[:, 2],
#         b1[:, 3],
#         b1[:, 4],
#         b1[:, 5],
#     )
#     b2_x1, b2_y1, b2_z1, b2_x2, b2_y2, b2_z2 = (
#         b2[:, 0],
#         b2[:, 1],
#         b2[:, 2],
#         b2[:, 3],
#         b2[:, 4],
#         b2[:, 5],
#     )
#     intersect_box_x1 = np.maximum(b1_x1, b2_x1)
#     intersect_box_y1 = np.maximum(b1_y1, b2_y1)
#     intersect_box_z1 = np.maximum(b1_z1, b2_z1)
#     intersect_box_x2 = np.minimum(b1_x2, b2_x2)
#     intersect_box_y2 = np.minimum(b1_y2, b2_y2)
#     intersect_box_z2 = np.minimum(b1_z2, b2_z2)

#     intersect_boxes = []
#     for i in range(N):
#         if (
#             intersect_box_x2[i] - intersect_box_x1[i] > 0
#             and intersect_box_y2[i] - intersect_box_y1[i] > 0
#             and intersect_box_z2[i] - intersect_box_z1[i] > 0
#         ):
#             if based_b1_correct_box:
#                 intersect_boxes.append(
#                     [
#                         intersect_box_x1[i] - b1_x1[0],
#                         intersect_box_y1[i] - b1_y1[0],
#                         intersect_box_z1[i] - b1_z1[0],
#                         intersect_box_x2[i] - b1_x1[0],
#                         intersect_box_y2[i] - b1_y1[0],
#                         intersect_box_z2[i] - b1_z1[0],
#                         b2[i, -1],
#                     ]
#                 )
#             else:
#                 intersect_boxes.append(
#                     [
#                         intersect_box_x1[i],
#                         intersect_box_y1[i],
#                         intersect_box_z1[i],
#                         intersect_box_x2[i],
#                         intersect_box_y2[i],
#                         intersect_box_z2[i],
#                         b2[i, -1],
#                     ]
#                 )
#     return intersect_boxes


# def split_3d_patch(annotation_file: str, save_folder: str, save_files: bool):
#     annotations: dict = load_json(annotation_file)
#     # 记录patch
#     patches = []
#     for No, value in annotations.items():
#         print(f"正在处理{No}...")
#         # 标注图像大小
#         shape = value["shape"]
#         # 边界框
#         boxes = []
#         for v in value["annotations"]:
#             if v["class"] == "fraction":
#                 boxes.append(v["location"] + [0])
#             elif v["class"] == "bladder":
#                 boxes.append(v["location"] + [1])
#         # 读取图像
#         if save_files:
#             ct = sitk.GetArrayFromImage(
#                 sitk.ReadImage(os.path.join("Files/resampled_FRI", No + "_rCT.nii.gz"))
#             )
#             suv = sitk.GetArrayFromImage(
#                 sitk.ReadImage(
#                     os.path.join(
#                         os.path.join("Files/resampled_FRI", No + "_rSUVbw.nii.gz")
#                     )
#                 )
#             )
#         # 分成 3D patch
#         num_z = math.ceil(shape[0] / 96.0)
#         num_y, dy = shape[1] // 96, (shape[1] % 96) // 2
#         num_x, dx = shape[2] // 96, (shape[2] % 96) // 2
#         for z in range(num_z):
#             for y in range(num_y):
#                 for x in range(num_x):
#                     patch = {"id": "_".join([No, str(x), str(y), str(z)])}
#                     if z != num_z - 1:
#                         patch["patch"] = [
#                             dx + x * 96,
#                             dy + y * 96,
#                             z * 96,
#                             dx + x * 96 + 96,
#                             dy + y * 96 + 96,
#                             z * 96 + 96,
#                         ]
#                     else:
#                         patch["patch"] = [
#                             dx + x * 96,
#                             dy + y * 96,
#                             shape[0] - 96,
#                             dx + x * 96 + 96,
#                             dy + y * 96 + 96,
#                             shape[0],
#                         ]
#                     patch["boxes"] = intersect_box(
#                         np.array([patch["patch"]]), np.array(boxes)
#                     )
#                     patches.append(patch)
#                     # 保存3d patch
#                     if save_files:
#                         print(f"=> 写入{x}-{y}-{z}")
#                         ct_patch = ct[
#                             patch["patch"][2] : patch["patch"][5],
#                             patch["patch"][1] : patch["patch"][4],
#                             patch["patch"][0] : patch["patch"][3],
#                         ]
#                         suv_patch = suv[
#                             patch["patch"][2] : patch["patch"][5],
#                             patch["patch"][1] : patch["patch"][4],
#                             patch["patch"][0] : patch["patch"][3],
#                         ]
#                         # 写在临时文件夹下面，再移入相应文件夹
#                         np.save(
#                             os.path.join(save_folder, patch["id"] + "_ct.npy"),
#                             np.array(ct_patch),
#                         )
#                         np.save(
#                             os.path.join(save_folder, patch["id"] + "_suv.npy"),
#                             np.array(suv_patch),
#                         )
#     return patches


# save_folder = "Files/patch"
# mkdir(save_folder)
# patches = split_3d_patch("Files/resampled_FRI/annotations.json", save_folder, False)

# train_patch_txt_path = os.path.join("Files", "train_patch.txt")
# val_patch_txt_path = os.path.join("Files", "val_patch.txt")
# output_folder = "model_data/image/patch"
# with open("Files/train.txt") as f:
#     train_lines = f.readlines()
# Nos = [line.split()[0] for line in train_lines]


# train_f = open(train_patch_txt_path, "w")
# val_f = open(val_patch_txt_path, "w")
# for patch in patches:
#     p_id = patch["id"]
#     file_no = p_id.split("_")[0]
#     line = " ".join([p_id] + [",".join(list(map(str, box))) for box in patch["boxes"]])
#     if file_no in Nos:
#         train_f.write(line + "\n")
#         train_f.flush()
#     else:
#         val_f.write(line + "\n")
#         val_f.flush()
# train_f.close()
# val_f.close()
