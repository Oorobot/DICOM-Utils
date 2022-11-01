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

from utils.dicom import (
    get_patient_info,
    get_SUVbw_in_GE,
    get_pixel_array,
    get_pixel_value,
    read_serises_image,
    resample,
)
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
#     # writer = sitk.ImageSeriesWriter()
#     # writer.SetFileNames(os.path.join(os.path.dirname(file), "Label_512.nii.gz"))
#     # writer.Execute(label512)
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


# 将标注数据进行转换json格式
labels = glob(os.path.join(folder_name, "*", "*Label*"))
for label in labels:
    folder = os.path.dirname(label)
    ct = os.path.join(folder, "CT.nii.gz")
    pet = os.path.join(folder, "SUVbw.nii.gz")


print(0)
