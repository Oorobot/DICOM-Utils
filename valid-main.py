import sys
from glob import glob

import cv2
import numpy as np
import SimpleITK as sitk
from PETCT import reg_data_valid

from utils import *

txt = open("valid.txt", "w")
sys.stdout = txt

# 分割标签数据校验
lung_slice = np.loadtxt("lung_slice.csv", np.uint32, delimiter=",", usecols=(1, 2))
seg_files = glob("PET-CT/*/*.nii.gz")

for file in seg_files:
    idx = int(file.split("\\")[-1].split(".")[0])
    start, end = lung_slice[idx - 1]
    img = sitk.ReadImage(file)
    length = img.GetSize()[-1]
    if length != end + 1 - start:
        print("not mathch, file: ", file)

# reg数据校验
reg_datas = glob("process/reg/*.npz")
reg_data_valid(reg_datas)
