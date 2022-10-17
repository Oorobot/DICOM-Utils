import os
import shutil
from glob import glob

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydicom
import SimpleITK as sitk
from mlxtend.evaluate import mcnemar, mcnemar_table

from utils.dicom import get_patient_info, get_SUVbw_in_GE, read_serises_image
from utils.metric import classification_metrics
from utils.utils import load_json, mkdir, save_json, to_pinyin

"""独立验证集, 提出的最佳模型与医生进行比较, 计算显著性水平p"""
# knee_ = pd.read_csv("Files/dicom/knee.csv")
# hip_ = pd.read_csv("Files/dicom/hip.csv")
# knee_label = knee_["label"].values
# hip_label = hip_["label"].values
# folds = ["fold 1", "fold 2", "fold 3", "fold 4", "fold 5"]
# doctors = ["D1", "D2", "D3"]
# print("Knee\nFold \tDoctor\tp\tchi2")
# for fold in folds:
#     for doctor in doctors:
#         tb = mcnemar_table(
#             y_target=knee_label,
#             y_model1=knee_[fold].values,
#             y_model2=knee_[doctor].values,
#         )
#         chi2, p = mcnemar(tb)
#         print("{}\t{}\t{:.4f}\t{:.4f}".format(fold, doctor, p, chi2))
# print("Hip\nFold \tDoctor\tp\tchi2")
# for fold in folds:
#     for doctor in doctors:
#         tb = mcnemar_table(
#             y_target=hip_label,
#             y_model1=hip_[fold].values,
#             y_model2=hip_[doctor].values,
#         )
#         chi2, p = mcnemar(tb)
#         print("{}\t{}\t{:.4f}\t{:.4f}".format(fold, doctor, p, chi2))


"""独立验证集, 制表给医生评估, 计算分类指标"""
# knee_ = pd.read_csv("Files/dicom/knee.csv")
# hip_ = pd.read_csv("Files/dicom/hip.csv")
# knee_label = knee_["label"].values
# hip_label = hip_["label"].values

# col_names = ["D1", "D2", "D3", "fold 1", "fold 2", "fold 3", "fold 4", "fold 5"]
# print("Knee")
# print("Doctor\t Acc\t Spec\t Sen\t F1\t PPV\t NPV")
# for col_name in col_names:
#     d = knee_[col_name].values
#     result = classification_metrics(knee_label, d)
#     print(
#         "{}  \t {:.4f}\t {:.4f}\t {:.4f}\t {:.4f}\t {:.4f}\t {:.4f}".format(
#             col_name, *result[1:-1]
#         )
#     )
# print("Hip")
# print("Doctor\t Acc\t Spec\t Sen\t F1\t PPV\t NPV")
# for col_name in col_names:
#     d = hip_[col_name].values
#     result = classification_metrics(hip_label, d)
#     print(
#         "{}  \t {:.4f}\t {:.4f}\t {:.4f}\t {:.4f}\t {:.4f}\t {:.4f}".format(
#             col_name, *result[1:-1]
#         )
#     )


"""统计已有的PET-FRI的病人信息"""
# base_path = "Data/PET-FRI&TPB-CT/PET-FRI/NormalData"
# PET_FRI_dirs = os.listdir(base_path)

# PET_FRI = pd.read_excel("Data/PET-FRI&TPB-CT/PET-FRI/PET-FRI.xlsx", "FRI")
# num = 0
# num_female = 0
# num_male = 0
# age = []
# num_infection = 0
# num_noninfection = 0
# for dir in PET_FRI_dirs:
#     no = int(dir[0:3])
#     # path = os.path.join(base_path, dir)
#     query_info = PET_FRI.query(f"No=={int(dir[:3])}")
#     age.append(int(query_info["Age"].values[0]))
#     gender = query_info["Gender"].values
#     if gender[0] == "Female":
#         num_female = num_female + 1
#     else:
#         num_male = num_male + 1
#     if query_info["Final_diagnosis"].values[0] == "T":
#         num_infection = num_infection + 1
#     else:
#         num_noninfection = num_noninfection + 1

#     num += 1

# print(
#     "num: ",
#     num,
#     ",num_female: ",
#     num_female,
#     num_female / num,
#     ",num_male: ",
#     num_male,
#     num_male / num,
#     ",num_infection: ",
#     num_infection,
#     num_infection / num,
#     ",num_noninfection: ",
#     num_noninfection,
#     num_noninfection / num,
# )
# print("age", np.mean(age), np.max(age), np.min(age))


"""统计PET-CT中存在肺结节标注的病人信息"""
# SEGMENTATION_FILES = glob("Data/PET-CT/*/*.nii.gz")
# LUNG_SLICE = pd.read_excel("Data/PET-CT/PET-CT.xlsx", "Sheet1")


# files = glob("Files/PETCT/*.npz")
# no_list = [int(os.path.basename(f)[0:3]) for f in files]
# no_list = list(set(no_list))
# result = LUNG_SLICE.query("编号 in @no_list")
# ## 统计男女个数，以及平均年龄
# age = result["年龄"].values
# age = [int(a[0:-2]) for a in age]
# print("【年龄】平均值: ", np.mean(age), " 最小值: ", np.min(age), " 最大值: ", np.max(age))
# sex = result["性别"].value_counts()
# print("【性别】", sex)


"""绘制PET-CT中 SUVmax, SUVmean, SUVmin 直方图"""
# plt.figure(figsize=(19.2, 10.8), dpi=100)
# files = glob("Files/PETCT_20221001/*.npz")
# suvmax = []
# suvmean = []
# suvmin = []
# for file in files:
#     data = np.load(file)
#     suvmax.append(data["SUVmax"])
#     suvmean.append(data["SUVmean"])
#     suvmin.append(data["SUVmin"])
# bins = [0, 1, 2, 3, 4, 5, 10, 20, 30, 50, 70]
# height = [np.histogram(suv, bins)[0] for suv in [suvmax, suvmean, suvmin]]
# left, n = np.arange(len(bins) - 1), len(height)
# ax = plt.subplot(111)
# colors = ["#63b2ee", "#76da91", "#f8cb7f"]
# labels = ["SUVmax", "SUVmean", "SUVmin"]
# # colors = ax._get_lines.color_cycle
# for j, h in enumerate(height):
#     b = ax.bar(
#         left + (j + 0.5) * 1.0 / n, h, width=1.0 / n, color=colors[j], label=labels[j]
#     )
#     ax.bar_label(b)
# ax.legend()
# ax.set_xticks(np.arange(0, len(bins)))
# ax.set_xticklabels(map(str, bins))
# ax.set_ylabel("Number")
# ax.set_xlabel("Standard Update Value")
# plt.show()


"""将骨三相CT中的存在缺失数据的文件夹和问题数据移至 problem_data 中"""

# TPB_CT = "F:\\ThreePhaseBone-CT\\data"
# TPB_CT_ = "F:\\ThreePhaseBone-CT\\problem_data"

# sub_dirs = os.listdir(TPB_CT)

# for sub_dir in sub_dirs:
#     sub_path = os.path.join(TPB_CT, sub_dir)
#     sub_sub_dirs = os.listdir(sub_path)
#     if len(sub_sub_dirs) == 0:
#         shutil.move(sub_path, os.path.join(TPB_CT_, sub_dir))
#     else:
#         sub_sub_dir = os.path.join(sub_path, "3.75")
#         if not os.listdir(sub_sub_dir):
#             shutil.move(sub_path, os.path.join(TPB_CT_, sub_dir))


# problem_data_info = pd.read_excel("F:\\ThreePhaseBone-CT\\问题数据.xlsx")
# dir_name = problem_data_info["编号"][0:39]
# for d in dir_name:
#     src = os.path.join(TPB_CT, str(d))
#     if os.path.exists(src):
#         shutil.move(src, os.path.join(TPB_CT_, str(d)))


"""PET-FRI 找出问题数据和存在缺失的数据"""
# PET_FRI = "Data/PET-FRI/数据"
# PET_FRI_ = "Data/PET-FRI/问题数据"
# problem_data_info = pd.read_excel("Data/PET-FRI/PET-FRI问题数据.xlsx")
# dir_name = problem_data_info["No"]
# for d in dir_name:
#     src = os.path.join(PET_FRI, str(d))
#     if os.path.exists(src):
#         shutil.move(src, os.path.join(PET_FRI_, str(d)))
# PET_FRI = "Data/PET-FRI/数据"
# PET_FRI_ = "Data/PET-FRI/问题数据"
# sub_dirs = os.listdir(PET_FRI)
# for sub_dir in sub_dirs:
#     sub_path = os.path.join(PET_FRI, sub_dir)
#     # 查看每个病人下的CT或PET是否为空
#     sub_sub_dirs = os.listdir(sub_path)
#     for sub_sub_dir in sub_sub_dirs:
#         sub_sub_path = os.path.join(sub_path, sub_sub_dir)
#         if os.path.isdir(sub_sub_path) and not os.listdir(sub_sub_path):
#             shutil.move(sub_path, os.path.join(PET_FRI_, sub_dir))
#             print(sub_path)
#             break


"""读取PET-FRI和TPB-CT中文件夹名、检查日期、病人姓名"""
# PET_FRI = "Data/PET-FRI/数据"
# patient_info = pd.read_excel("Data/PET-FRI/PET-FRI.xlsx", "FRI")


# TPB_CT = "Data/ThreePhaseBone-CT/问题数据"
# patient_info = pd.read_excel("Data/ThreePhaseBone/ThreePhaseBone.xlsx")

# dirs = os.listdir(TPB_CT)
# xlxs = pd.DataFrame(
#     columns=[
#         "No",
#         "Folder",
#         "Datetime",
#         "Name",
#         "Sex",
#         "NameFromExcel",
#         "SexFromExcel",
#         "NameToPinyin",
#         "TimeFromExcel",
#     ]
# )
# i = 1
# for dir in dirs:
#     # 每个病人的数据, 子文件夹 CT、PET 中读取一个文件，主文件下读取所有文件
#     print(f"编号: {dir:>3}")
#     patient_dir = os.path.join(TPB_CT, dir)
#     sub_dirs = os.listdir(patient_dir)
#     for sub_dir in sub_dirs:
#         patient_sub_dir = os.path.join(patient_dir, sub_dir)
#         if os.path.isdir(patient_sub_dir):
#             filenames = os.listdir(patient_sub_dir)
#             if len(filenames) == 0:
#                 continue
#             dicom_filename = os.path.join(patient_sub_dir, filenames[0])
#             information = get_patient_info(dicom_filename)
#         else:
#             # 仅查看PET或者CT的文件中的信息
#             continue

#         # query_info = patient_info.query(f"No=={int(dir[:3])}")
#         # info = query_info[["Name", "Gender"]].values
#         query_info = patient_info.query(f"编号=={int(dir[:3])}")
#         info = query_info[["姓名", "性别", "检查日期"]].values
#         pinyin = to_pinyin(info[0][0])
#         xlxs.loc[i] = [
#             dir,
#             sub_dir,
#             information["Acquisition Date"],
#             information["Patient Name"],
#             information["Patient Sex"],
#             info[0][0],
#             info[0][1],
#             pinyin,
#             info[0][2],
#         ]
#         i = i + 1

# xlxs.to_excel(TPB_CT + ".xlsx", "Sheet1", index=False)


"""获取骨三相数据中的姓名性别影像学ID"""
# folder = r"C:\Users\admin\Desktop\Data\ThreePhaseBone-CT\NormalData"
# dirs = os.listdir(folder)
# excel_file = r"C:\Users\admin\Desktop\Data\ThreePhaseBone\ThreePhaseBone.xlsx"
# tags = {
#     0x00100010: "Patient Name",
#     0x00100020: "Patient ID",
#     0x00100040: "Patient Sex",
#     0x00080014: "Instance Creator UID",
#     0x00080016: "SOP Class UID",
#     0x00080018: "SOP Instance UID",
#     0x0020000D: "Study Instance UID",
#     0x0020000E: "Series Instance UID",
#     0x00200052: "Frame of Reference UID",
# }
# tpb = pd.read_excel(excel_file)

# new_excel = pd.DataFrame(columns=["姓名", "性别", *list(tags.values())])
# for dir in dirs:
#     no = int(dir)
#     name_sex = tpb.query(f"编号 == {no}")[["姓名", "性别"]].values
#     file = os.path.join(folder, dir, "ImageFileName.dcm")
#     infomation = get_patient_info(file, tags)
#     infomation["姓名"] = name_sex[0, 0].strip()
#     infomation["性别"] = name_sex[0, 1].strip()
#     new_excel.loc[len(new_excel)] = infomation
# new_excel.to_excel("TPB_ID.xlsx")

"""将PET图像中的矩阵值转换为SUV值"""
path = "D:\\Desktop\\Data\\PET-FRI\\NormalData"

# 移动文件

result_path = "D:\\Desktop\\result"
mkdir(result_path)


c = os.listdir(path)[-210:-100] + os.listdir(path)[-50:]
choosed = os.listdir(path)[-100:-50]
for _ in c:
    _path = os.path.join(path, _)
    _result = os.path.join(result_path, _)
    mkdir(_result)
    for _p in os.listdir(_path):
        __path = os.path.join(_path, _p)
        if not os.path.isdir(__path):
            shutil.copy(__path, _result)
print(0)
# PET_dirs = os.listdir(path)
# for dir in PET_dirs:
#     files = glob(os.path.join(path, dir, "PET", "*.dcm"))
#     pet = read_serises_image(files)
#     pixel_array = sitk.GetArrayFromImage(pet)
#     flag = pixel_array.dtype == np.int16
#     if flag:
#         print("---skip---", dir)
#         continue
#     suvbw = np.zeros_like(pixel_array, dtype=np.float64)
#     for i in range(pixel_array.shape[0]):
#         suvbw[i] = get_SUVbw_in_GE(pixel_array[i], files[i])
#         for j in range(128):
#             for k in range(128):
#                 pet[k, j, i] = suvbw[i, j, k]
#     print("done---", dir)
#     sitk.WriteImage(pet, os.path.join(path, dir, "PET", "suvbw.nii.gz"))

# left_dirs = [
#     "120",
#     "270",
#     "327",
#     "370",
#     "393",
#     "453",
#     "456",
#     "457",
#     "489",
#     "592",
#     "600",
#     "634",
#     "725",
# ]
# for dir in left_dirs:
#     files = glob(os.path.join(path, dir, "PET", "*.dcm"))
#     pet = read_serises_image(files)
#     pixel_array = sitk.GetArrayFromImage(pet)
#     suvbw = np.zeros_like(pixel_array, dtype=np.float64)
#     for i in range(pixel_array.shape[0]):
#         suvbw[i] = get_SUVbw_in_GE(pixel_array[i], files[i])
#     suvbw_image = sitk.GetImageFromArray(suvbw)
#     suvbw_image.SetDirection(pet.GetDirection())
#     suvbw_image.SetOrigin(pet.GetOrigin())
#     suvbw_image.SetSpacing(pet.GetSpacing())
#     # suvbw_image.SetPixelAsComplexFloat64(())
#     sitk.WriteImage(suvbw_image, os.path.join(path, dir, "PET", "suvbw.nii.gz"))
#     print(f"{dir} done.")

# dirs = os.listdir(TPB_CT)
# xlxs = pd.DataFrame(
#     columns=[
#         "No",
#         "Folder",
#         "Datetime",
#         "Name",
#         "Sex",
#         "NameFromExcel",
#         "SexFromExcel",
#         "NameToPinyin",
#         "TimeFromExcel",
#     ]
# )
# i = 1
# for dir in dirs:
#     # 每个病人的数据, 子文件夹 CT、PET 中读取一个文件，主文件下读取所有文件
#     print(f"编号: {dir:>3}")
#     patient_dir = os.path.join(TPB_CT, dir)
#     sub_dirs = os.listdir(patient_dir)
#     for sub_dir in sub_dirs:
#         patient_sub_dir = os.path.join(patient_dir, sub_dir)
#         if os.path.isdir(patient_sub_dir):
#             filenames = os.listdir(patient_sub_dir)
#             if len(filenames) == 0:
#                 continue
#             dicom_filename = os.path.join(patient_sub_dir, filenames[0])
#             information = get_patient_info(dicom_filename)
#         else:
#             # 仅查看PET或者CT的文件中的信息
#             continue

#         # query_info = patient_info.query(f"No=={int(dir[:3])}")
#         # info = query_info[["Name", "Gender"]].values
#         query_info = patient_info.query(f"编号=={int(dir[:3])}")
#         info = query_info[["姓名", "性别", "检查日期"]].values
#         pinyin = to_pinyin(info[0][0])
#         xlxs.loc[i] = [
#             dir,
#             sub_dir,
#             information["Acquisition Date"],
#             information["Patient Name"],
#             information["Patient Sex"],
#             info[0][0],
#             info[0][1],
#             pinyin,
#             info[0][2],
#         ]
#         i = i + 1

# xlxs.to_excel(TPB_CT + ".xlsx", "Sheet1", index=False)


""" 绘制肺结节最大直方图 """
# pulmonary_nodules = load_json("pulmonary_nodules.json")
# distance = []
# for CT_no, value in pulmonary_nodules.items():

#     for slice_no, v in value.items():
#         if slice_no == "Spacing":
#             continue
#         else:
#             distance.append(v["distance"])
# height, bins = np.histogram(distance, bins=np.arange(np.max(distance) + 1))

# plt.figure(figsize=(19.2, 10.8), dpi=100)
# ax = plt.subplot(1, 1, 1)
# left = np.arange(len(bins) - 1)

# b = ax.bar(left + 0.5, height, 0.8)
# ax.bar_label(b)
# ax.set_xticks(np.arange(0, len(bins)))
# ax.set_xticklabels(map(str, bins.astype(np.uint16)))
# ax.tick_params(labelsize=8)
# ax.set_ylabel("Number")
# ax.set_xlabel("Maximum Diameter")

# file = np.load("Files/PETCT_20221001/076_CT335_00.npz")["mask"]
# cv2.imwrite(
#     "t1.jpg",
#     file * 255,
#     [int(cv2.IMWRITE_JPEG_QUALITY), 100],
# )
