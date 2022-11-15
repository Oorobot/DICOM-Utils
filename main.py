import datetime
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

from utils.dicom import (get_3D_annotation, get_patient_info, get_pixel_array,
                         get_pixel_value, get_SUVbw_in_GE, read_serises_image,
                         resample, resample_spacing)
from utils.metric import classification_metrics
from utils.utils import delete, load_json, mkdir, rename, save_json, to_pinyin

folder_name = "D:/admin/Desktop/Data/PET-FRI/NormalData"
src = "./Files"

""""""
# dirs = []
# with zipfile.ZipFile("D:/admin/Desktop/795-985.zip") as zip:
#     filelist = zip.namelist()
#     for file in filelist:
#         if ".nii.gz" in file and "suv" not in file:
#             dir_name = os.path.dirname(file)
#             dirs.append(dir_name)
#             # 提取压缩包里面的标注数据
#             zip.extract(file, "Files")
#             shutil.move(os.path.join(src, file), os.path.join(folder_name, file))
#             shutil.rmtree(os.path.join(src, dir_name))
# 删除掉未校验的标注数据
# for d in dirs:
#     folder = os.path.join(folder_name, d)
#     if not os.path.exists(folder):
#         continue
#     files = os.listdir(folder)
#     for file in files:
#         if ".nii.gz" in file and "SUVbw" not in file and "CT" not in file:
#             delete(os.path.join(folder, file))


# 将标注数据resize到CT大小
# files = glob("D:/Desktop/Data/PET-FRI/NormalData/*/U*.nii.gz")
# for file in files:
#     image = sitk.ReadImage(file)
#     cts = glob(os.path.join(os.path.dirname(file), "CT") + "/*dcm")
#     CT = read_serises_image(cts)
#     pets = glob(os.path.join(os.path.dirname(file), "PET") + "/*dcm")
#     PET = read_serises_image(pets)
#     label512 = resample(image, CT, True)
#     sitk.WriteImage(label512, os.path.join(os.path.dirname(file), "Label_512.nii.gz"))
# print(0)


# 数据处理
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


# 查重
# list = []
# excel_info = pd.read_excel("D:/Desktop/Data/PET-FRI/PET-FRI.xlsx", "FRI")
# for dir in os.listdir(folder_name):
#     cts = glob(os.path.join(folder_name, dir, "CT", "*"))
#     pets = glob(os.path.join(folder_name, dir, "PET", "*"))
#     ct_info = get_patient_info(cts[0])
#     pet_info = get_patient_info(pets[0])
#     info = excel_info.query(f"No == {int(dir[:3])}")[["Name"]].values.tolist()
#     list.append(
#         "-".join(["CT", str(ct_info["Patient Name"]), str(ct_info["Acquisition Date"])])
#     )
#     list.append(
#         "-".join(
#             ["PET", str(pet_info["Patient Name"]), str(pet_info["Acquisition Date"])]
#         )
#     )
#     list += info[0]

# a = pd.DataFrame(list).value_counts()
# print(a[:21])
# print(0)


# 合并两个Excel表格
# excel1 = pd.read_excel("D:/admin/Desktop/Data/PET-FRI/PET-FRI.xlsx", "FRI-unique")
# excel2 = pd.read_excel("D:/admin/Desktop/Data/PET-FRI/PET-FRI.xlsx", "PET-all")
# excel3 = pd.merge(excel1, excel2, how="left", on=["Name", "影像学表现", "影像学诊断"])
# excel3.to_excel("D:/admin/Desktop/Data/PET-FRI/result.xlsx")


# 查询病人数据跟DICOM的获取时间是否一致
# list = []
# excel_info = pd.read_excel("D:/admin/Desktop/Data/PET-FRI/result.xlsx")
# log = open("./Files/result.txt", "a", encoding="utf-8")
# for dir in os.listdir(folder_name):
#     cts = glob(os.path.join(folder_name, dir, "CT", "*"))
#     pets = glob(os.path.join(folder_name, dir, "PET", "*"))
#     ct_info = get_patient_info(cts[0])
#     pet_info = get_patient_info(pets[0])
#     info = excel_info.query(f"No_FRI == {int(dir[:3])}")[
#         ["Name", "Date"]
#     ].values.tolist()
#     list.append(
#         "-".join(["CT", str(ct_info["Patient Name"]), str(ct_info["Acquisition Date"])])
#     )
#     list.append(
#         "-".join(
#             ["PET", str(pet_info["Patient Name"]), str(pet_info["Acquisition Date"])]
#         )
#     )

#     name_pinyin = to_pinyin(info[0][0])
#     date = (
#         info[0][1].strftime("%Y%m%d") if not isinstance(info[0][1], str) else info[0][1]
#     )
#     if (
#         # name_pinyin != str(ct_info["Patient Name"])
#         # or name_pinyin != str(pet_info["Patient Name"])
#         date != str(ct_info["Acquisition Date"])
#         or date != str(pet_info["Acquisition Date"])
#     ):
#         log.write(
#             "{:<5}-名字：{:<10}{:<18}{:<18}{:<18}，日期：{:<10}{:<10}{:<10}\n".format(
#                 dir[:3],
#                 info[0][0],
#                 to_pinyin(info[0][0]),
#                 str(ct_info["Patient Name"]),
#                 str(pet_info["Patient Name"]),
#                 info[0][1].strftime("%Y%m%d")
#                 if not isinstance(info[0][1], str)
#                 else info[0][1],
#                 str(ct_info["Acquisition Date"]),
#                 str(pet_info["Acquisition Date"]),
#             )
#         )
#         log.flush()
# log.close


# # 移动文件
# files = glob(os.path.join(folder_name, "*.nii.gz"))
# for file in files:
#     filename = os.path.basename(file)
#     dirname = filename.split("_")[0]
#     print(file, "->", os.path.join(folder_name, dirname, filename))
#     rename(file, os.path.join(folder_name, dirname, filename))
# files = glob(os.path.join(folder_name, "*", "*.nii.gz"))
# for dirname in os.listdir(folder_name):
#     files = glob(os.path.join(folder_name, dirname, "*.nii.gz"))
#     if len(files) == 2:
#         for file in files:
#             filename = os.path.basename(file)
#             shutil.copyfile(file, os.path.join(src, "FRI", filename))
# temp_dir = os.path.join(src, ".temp")
# mkdir(os.path.join(src, ".temp"))
# with zipfile.ZipFile("D:/admin/Desktop/795-985.zip") as zip:
#     filelist = zip.namelist()
#     for file in filelist:
#         if "Untitled" in file:
#             zip.extract(file, temp_dir)
#             filename = os.path.join(temp_dir, file)
#             label = sitk.ReadImage(filename)
#             dirname = os.path.dirname(file)
#             ct_files = glob(os.path.join(folder_name, dirname, "CT", "*dcm"))
#             CT = read_serises_image(ct_files)
#             resampled_label = resample(label, CT, True)
#             sitk.WriteImage(
#                 resampled_label, os.path.join(src, "FRI", f"{dirname}_CT_Label.nii.gz")
#             )

# zips = ["D:/admin/Desktop/2.zip", "D:/admin/Desktop/3.zip"]
# files = glob("D:/admin/Desktop/2/*")
# for file in files:

#     dirname = os.path.basename(file)[:3]
#     if os.path.exists(os.path.join(src, "FRI", f"{dirname}_CT_Label.nii.gz")):
#         continue

#     ct_files = glob(os.path.join(folder_name, dirname, "CT", "*dcm"))

#     if len(ct_files) == 0:
#         continue

#     label = sitk.ReadImage(file)

#     CT = read_serises_image(ct_files)

#     resampled_label = resample(label, CT, True)

#     sitk.WriteImage(
#         resampled_label,
#         os.path.join(src, "FRI", f"{dirname}_CT_Label.nii.gz"),
#     )


# # 将标注数据进行转换json格式
# labels = glob(os.path.join(folder_name, "*", "*Label*"))
# annotations = {}
# for label in labels:

#     folder = os.path.dirname(label)
#     No = os.path.basename(folder)
#     ct = os.path.join(folder, "CT.nii.gz")
#     pet = os.path.join(folder, "SUVbw.nii.gz")
#     # 获取矩阵数据
#     label_image = sitk.ReadImage(label)
#     label_array = sitk.GetArrayFromImage(label_image)
#     # 获取标注数据的类数量
#     classes_num = int(np.max(label_array))
#     # 寻找每个类的标注
#     annotation = []
#     for i in range(1, classes_num + 1):
#         class_label_array = np.where(label_array != i, 0, 1)
#         for slice in class_label_array:
#             if 0 != np.max(slice):
#                 contours, _ = cv2.findContours(
#                     slice.astype(np.uint8),
#                     cv2.RETR_LIST,
#                     cv2.CHAIN_APPROX_SIMPLE,
#                 )
#                 for contour in contours:
#                     contour = np.squeeze(contour)
#                     x1, y1 = contour[0]
#                     x2, y2 = contour[2]

#                     cs, _ = cv2.findContours(
#                         class_label_array[:, :, x1].astype(np.uint8),
#                         cv2.RETR_LIST,
#                         cv2.CHAIN_APPROX_SIMPLE,
#                     )
#                     for c in cs:
#                         c = np.squeeze(c)
#                         y1_, z1 = c[0]
#                         y2_, z2 = c[2]
#                         if y1 == y1_ and y2 == y2_:
#                             annotation.append(
#                                 {
#                                     "class": class_name[i],
#                                     "location": [x1, y1, z1, x2, y2, z2],
#                                 }
#                             )
#                             annotation[-1]["location"] = [
#                                 int(_) for _ in annotation[-1]["location"]
#                             ]
#                             # 清除此标注区域数据
#                             class_label_array[z1 : z2 + 1, y1 : y2 + 1, x1 : x2 + 1] = 0

#     annotations[No] = annotation

#     save_json("./Files/annotations.json", annotations)


class_name = {1: "fraction", 2: "bladder", 3: "other"}
labels = glob(os.path.join(folder_name, "628", "*Label*"))
annotations = load_json("./Files/resampled_FRI/annotations.json")
output_folder = "./Files/resampled_FRI"
mkdir(output_folder)
# for label in labels:
#     folder = os.path.dirname(label)
#     no = os.path.basename(folder)
#     ct = os.path.join(folder, f"{no}_CT.nii.gz")
#     pet = os.path.join(folder, f"{no}_SUVbw.nii.gz")

#     # 读取CT数据
#     ct_image = sitk.ReadImage(ct)
#     # ct_array = sitk.GetArrayFromImage(ct_image)
#     suvbw_image = sitk.ReadImage(pet)
#     label_image = sitk.ReadImage(label)
#     # 进行重采样到均为 spacing  1x1x1 cm
#     resampled_ct = resample_spacing(ct_image)
#     resampled_suvbw = resample(suvbw_image, resampled_ct)
#     resampled_label = resample(label_image, resampled_ct, True)

#     # 写入重采样的文件
#     sitk.WriteImage(resampled_ct, os.path.join(output_folder, f"{no}_rCT.nii.gz"))
#     sitk.WriteImage(resampled_suvbw, os.path.join(output_folder, f"{no}_rSUVbw.nii.gz"))
#     sitk.WriteImage(resampled_label, os.path.join(output_folder, f"{no}_rLabel.nii.gz"))

#     """记录标注数据"""
#     resample_label_array = sitk.GetArrayFromImage(resampled_label)
#     classes, locations = get_3D_annotation(resample_label_array)
#     for c, l in zip(classes, locations):
#         annotations[no] = [
#             {"class": class_name[c], "location": l} for c, l in zip(classes, locations)
#         ]

# label = sitk.ReadImage(os.path.join(output_folder, "097_rLabel.nii.gz"))
# label_array = sitk.GetArrayFromImage(label)
# """记录标注数据"""
# classes, locations = get_3D_annotation(label_array)
# # annotations[no] = []
# for c, l in zip(classes, locations):
#     annotations["628"] = [
#         {"class": class_name[c], "location": l} for c, l in zip(classes, locations)
#     ]

# save_json(os.path.join(output_folder, "annotations.json"), annotations)


suvs = ["Files/resampled_FRI/633_rSUVbw.nii.gz"]

for f in suvs:
    no = os.path.basename(f).split("_")[0]
    if int(no) < 633:
        continue
    array = sitk.GetArrayFromImage(sitk.ReadImage(f))
    array = np.array(array, dtype=np.float32)
    filename = os.path.basename(f).split(".")[0]
    np.save(os.path.join("./Files", filename), array)
