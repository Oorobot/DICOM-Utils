from glob import glob
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# # 存在肺结节标注的病人数据
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

# # 绘制 SUVmax, SUVmean, SUVmin 直方图
# plt.figure(figsize=(19.2, 10.8), dpi=100)
# files = glob("Files/PETCT/*.npz")
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
#     # for a, b in zip(left + (j + 0.5) * 1.0 / n, h):
#     #     ax.text(a, b + 1, b, ha="center", va="bottom")
# ax.legend()
# ax.set_xticks(np.arange(0, len(bins)))
# ax.set_xticklabels(map(str, bins))
# ax.set_ylabel("Number")
# ax.set_xlabel("Standard Update Value")
# plt.show()


# 整理 PET-FRI 中的病人数据
base_path = "Data/PET-FRI&TPB-CT/PET-FRI/NormalData"
PET_FRI_dirs = os.listdir(base_path)

PET_FRI = pd.read_excel("Data/PET-FRI&TPB-CT/PET-FRI/PET-FRI.xlsx", "FRI")
num = 0
num_female = 0
num_male = 0
age = []
num_infection = 0
num_noninfection = 0
for dir in PET_FRI_dirs:
    no = int(dir[0:3])
    # path = os.path.join(base_path, dir)
    query_info = PET_FRI.query(f"No=={int(dir[:3])}")
    age.append(int(query_info["Age"].values[0]))
    gender = query_info["Gender"].values
    if gender[0] == "Female":
        num_female = num_female + 1
    else:
        num_male = num_male + 1
    if query_info["Final_diagnosis"].values[0] == "T":
        num_infection = num_infection + 1
    else:
        num_noninfection = num_noninfection + 1

    num += 1

print(
    "num: ",
    num,
    ",num_female: ",
    num_female,
    num_female / num,
    ",num_male: ",
    num_male,
    num_male / num,
    ",num_infection: ",
    num_infection,
    num_infection / num,
    ",num_noninfection: ",
    num_noninfection,
    num_noninfection / num,
)
print("age", np.mean(age), np.max(age), np.min(age))
