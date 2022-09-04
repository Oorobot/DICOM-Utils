from glob import glob
import os
import pandas as pd
import numpy as np

# 存在肺结节标注的病人数据
SEGMENTATION_FILES = glob("Data/PET-CT/*/*.nii.gz")
LUNG_SLICE = pd.read_excel("Data/PET-CT/PET-CT.xlsx", "Sheet1")


files = glob("Files/PETCT/*.npz")
no_list = [int(os.path.basename(f)[0:3]) for f in files]
no_list = list(set(no_list))
result = LUNG_SLICE.query("编号 in @no_list")
## 统计男女个数，以及平均年龄
age = result["年龄"].values
age = [int(a[0:-2]) for a in age]
print("【年龄】平均值: ", np.mean(age), " 最小值: ", np.min(age), " 最大值: ", np.max(age))
sex = result["性别"].value_counts()
print("【性别】", sex)
