import os
import shutil
import pandas as pd


# 将骨三相CT中的存在缺失数据的文件夹移至 problem_data 中

# TPB_CT = "F:\\ThreePhaseBone-CT\\data"
# TPB_CT_ = "F:\\ThreePhaseBone-CT\\problem_data"

# sub_dirs = os.listdir(TPB_CT)

# # 找出来的存在缺失的问题数据
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


# # PET-FRI 找出问题数据
# PET_FRI = "Data/PET-FRI/数据"
# PET_FRI_ = "Data/PET-FRI/问题数据"
# problem_data_info = pd.read_excel("Data/PET-FRI/PET-FRI问题数据.xlsx")
# dir_name = problem_data_info["No"]
# for d in dir_name:
#     src = os.path.join(PET_FRI, str(d))
#     if os.path.exists(src):
#         shutil.move(src, os.path.join(PET_FRI_, str(d)))

# # 找出PET-FRI中存在缺失的数据
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


# # 文件夹重命名
# TPB_CT = "Data/ThreePhaseBone-CT/数据"
# TPB_CT_ = "Data/ThreePhaseBone-CT/问题数据"
# PET_FRI = "Data/PET-FRI/数据"
# PET_FRI_ = "Data/PET-FRI/问题数据"

# for path in [TPB_CT, TPB_CT_, PET_FRI, PET_FRI_]:
#     dirs = os.listdir(path)
#     for dir in dirs:
#         src = os.path.join(path, dir)
#         dir = dir.zfill(3)
#         tar = os.path.join(path, dir)
#         shutil.move(src, tar)
#         print(src, " => ", tar)


# 读取PET-FRI中文件夹名、检查日期、病人姓名
import pydicom
import pandas as pd
from xpinyin import Pinyin

# PET_FRI = "Data/PET-FRI/数据"
# patient_info = pd.read_excel("Data/PET-FRI/PET-FRI.xlsx", "FRI")


TPB_CT = "Data/ThreePhaseBone-CT/问题数据"
patient_info = pd.read_excel("Data/ThreePhaseBone/ThreePhaseBone.xlsx")

dirs = os.listdir(TPB_CT)
p = Pinyin()
xlxs = pd.DataFrame(
    columns=[
        "No",
        "Folder",
        "Datetime",
        "Name",
        "Sex",
        "NameFromExcel",
        "SexFromExcel",
        "NameToPinyin",
        "TimeFromExcel",
    ]
)
i = 1
for dir in dirs:
    # 每个病人的数据, 子文件夹 CT、PET 中读取一个文件，主文件下读取所有文件
    print(f"编号: {dir:>3}")
    patient_dir = os.path.join(TPB_CT, dir)
    sub_dirs = os.listdir(patient_dir)
    for sub_dir in sub_dirs:
        patient_sub_dir = os.path.join(patient_dir, sub_dir)
        if os.path.isdir(patient_sub_dir):
            filenames = os.listdir(patient_sub_dir)
            if len(filenames) == 0:
                continue
            dicom_filename = os.path.join(patient_sub_dir, filenames[0])
            file = pydicom.dcmread(dicom_filename)
        else:
            # 仅查看PET或者CT的文件中的信息
            continue

        try:
            AcqusitionDate = file.AcquisitionDate
        except Exception as e:
            AcqusitionDate = ""
        PatientName = file.PatientName.family_name
        PatientSex = file.PatientSex
        # query_info = patient_info.query(f"No=={int(dir[:3])}")
        # info = query_info[["Name", "Gender"]].values
        query_info = patient_info.query(f"编号=={int(dir[:3])}")
        info = query_info[["姓名", "性别", "检查日期"]].values
        pinyin = p.get_pinyin(info[0][0].replace("\t", ""), " ", convert="upper")
        if pinyin != PatientName:
            print(
                f"{sub_dir:>20} - {AcqusitionDate:>15} - {PatientName:>15} - {PatientSex:>2} - {info[0][0]:>8} - {info[0][1]:>8} - {pinyin:>15} - {info[0][2]:>8} "
            )
        xlxs.loc[i] = [
            dir,
            sub_dir,
            AcqusitionDate,
            PatientName,
            PatientSex,
            info[0][0],
            info[0][1],
            pinyin,
            info[0][2],
        ]
        i = i + 1

xlxs.to_excel(TPB_CT + ".xlsx", "Sheet1", index=False)
