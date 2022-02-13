import datetime
import math
from glob import glob
from typing import List

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydicom
from xpinyin import Pinyin

from dicom import get_pixel_value
from utils import save_json


def validate_PETCT_regression_data(filelist: List[str], txt: str):
    txt_file = open(txt, "w")
    max = 0.0
    min = np.inf

    file_list1 = []
    file_list2 = []

    for file in filelist:

        data = np.load(file)
        hu = data["HU"]
        seg = data["segmentation"]
        suvmax = data["max"]
        suvmean = data["mean"]
        suvmin = data["min"]

        contours, _ = cv2.findContours(
            seg.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE
        )
        if len(contours) > 1:
            file_list1.append(file)

        if (
            np.isnan(hu).any()
            and np.isinf(hu).any()
            and np.isnan(seg).any()
            and np.isinf(seg).any()
        ):
            file_list2.append(file)
        print(
            "filename: %s, max: %f, min: %f, mean: %f."
            % (file, suvmax, suvmin, suvmean),
            file=txt_file,
        )
        if suvmax > max:
            max = suvmax
        if suvmin < min:
            min = suvmin
    print(
        "the max of SUVbw max: ", max, file=txt_file,
    )
    print(
        "the min of SUVbw min: ", min, file=txt_file,
    )
    print(
        "--- THESE FILES HAVE MORE THAN 2 CONTOUR ---\n", file_list1, file=txt_file,
    )
    print(
        "--- THESE FILES HAS INF OR NAN ---\n", file_list2, file=txt_file,
    )
    txt_file.close()


def validate_patient_information(flows: List[str], xlsx_name: str):
    xlsx = pd.read_excel(xlsx_name, sheet_name="Sheet1", usecols=[0, 1, 2])

    p = Pinyin()
    values = xlsx.values
    for flow in flows:
        img = pydicom.dcmread(flow)
        img_arr = get_pixel_value(flow)
        index = flow.split("\\")[1].split("_")[0]
        AcqusitionDate = img.AcquisitionDate
        PatientName = img.PatientName.family_name
        PatientName = PatientName.replace(" ", "")

        xlsx_value = values[int(index) - 1]
        xlsx_name = xlsx_value[2].strip("\t")
        xlsx_date = xlsx_value[1].strip("\t")
        xlsx_date = datetime.strptime(xlsx_date, "%Y-%m-%d").strftime("%Y%m%d")

        xlsx_pinyin = p.get_pinyin(xlsx_name, splitter="", convert="upper")
        if (
            xlsx_pinyin != PatientName
            or AcqusitionDate != xlsx_date
            or img_arr.shape[1:] != (128, 128)
            or img_arr.shape[0] < 25
        ):
            print(
                "filename: %s --- patient name: %s(%s-%s), date: %s-%s, date shape: %s."
                % (
                    flow,
                    xlsx_name,
                    xlsx_pinyin,
                    PatientName,
                    xlsx_date,
                    AcqusitionDate,
                    str(img_arr.shape),
                )
            )


# # 分割标签数据校验
# lung_slice = np.loadtxt("lung_slice.csv", np.uint32, delimiter=",", usecols=(1, 2))
# seg_files = glob("PET-CT/*/*.nii.gz")

# for file in seg_files:
#     idx = int(file.split("\\")[-1].split(".")[0])
#     start, end = lung_slice[idx - 1]
#     img = sitk.ReadImage(file)
#     length = img.GetSize()[-1]
#     if length != end + 1 - start:
#         print("not mathch, file: ", file)

# # reg数据校验
reg_data = glob("ProcessedData/regression/*.npz")
validate_PETCT_regression_data(reg_data, "valid.txt")

# regession 数据 按找 suvmax 进行划分, 画 suvmax, suvmin, suvmean 直方图
suvmax = []
suvmin = []
suvmean = []
max0_1 = []
max1_2 = []
max2_3 = []
max3_4 = []
max4_5 = []
max5_10 = []
max10_ = []
for file in reg_data:
    data = np.load(file)
    max = data["max"]
    mean = data["mean"]
    min = data["min"]
    suvmax.append(max)
    suvmean.append(mean)
    suvmin.append(min)
    if 0 <= max < 1:
        max0_1.append(file)
    elif 1 <= max < 2:
        max1_2.append(file)
    elif 2 <= max < 3:
        max2_3.append(file)
    elif 3 <= max < 4:
        max3_4.append(file)
    elif 4 <= max < 5:
        max4_5.append(file)
    elif 5 <= max < 10:
        max5_10.append(file)
    elif max >= 10:
        max10_.append(file)

suvmax = np.array(suvmax)
suvmin = np.array(suvmin)
suvmean = np.array(suvmean)


def hist(array: np.ndarray, save_path: str):

    plt.figure()
    plt.hist(
        array, math.ceil(np.max(array) - np.min(array)), (np.min(array), np.max(array)),
    )
    plt.savefig(save_path)
    plt.close()


hist(suvmax, "suvmax.png")
hist(suvmin, "suvmin.png")
hist(suvmean, "suvmean.png")

save_json(
    "ProcessedData/regression/data.json",
    {
        "0-1": max0_1,
        "1-2": max1_2,
        "2-3": max2_3,
        "3-4": max3_4,
        "4-5": max4_5,
        "5-10": max5_10,
        "10-": max10_,
    },
)

