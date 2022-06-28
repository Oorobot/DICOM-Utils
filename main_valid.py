import math
from datetime import datetime
from glob import glob
from typing import List

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydicom
import SimpleITK as sitk
from xpinyin import Pinyin

from utils.dicom import get_pixel_value


def validate_PETCT_regression_data(filelist: List[str]):
    log = open("validate.log", "a", encoding="UTF-8")
    log.write(
        "===> [{}] Validate {} PET-CT Regression Data.\n".format(
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"), len(filelist)
        )
    )
    log.flush()
    suv = []
    for filename in filelist:
        data = np.load(filename)
        hu, seg = data["HU"], data["segmentation"]
        suv.append([data["max"], data["mean"], data["min"]])
        contours, _ = cv2.findContours(
            seg.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE
        )
        if (len(contours) > 1) or (
            np.isnan(hu).any()
            and np.isinf(hu).any()
            and np.isnan(seg).any()
            and np.isinf(seg).any()
        ):
            log.write(
                "File Name: {0}, the number of contour: {1}, [MAX] HU: {2} - SEG: {3}, [MIN] HU: {4} - SEG: {5}.\n".format(
                    filename,
                    len(contours),
                    np.max(hu),
                    np.max(seg),
                    np.min(hu),
                    np.min(seg),
                )
            )
            log.flush()
    suv = np.reshape(np.concatenate(suv), (-1, 3))
    log.write(
        "[MAX] suvmax: {0:.4f}, suvmean: {1:.4f}, suvmin: {2:.4f} - [MIN] suvmax: {3:.4f}, suvmean: {4:.4f}, suvmin: {5:.4f}.\n".format(
            *np.max(suv, axis=0), *np.min(suv, axis=0)
        )
    )
    log.close()
    plt.figure()
    titles = ["SUVmax", "SUVmean", "SUVmin"]
    for i, array in enumerate(suv.T):
        plt.subplot(3, 1, i + 1)
        plt.hist(
            array,
            math.ceil(np.max(array) - np.min(array)),
            (np.min(array), np.max(array)),
        )
        plt.title(titles[i])
    plt.tight_layout(pad=0.5, h_pad=5.0, w_pad=5.0)
    plt.savefig("hist_suv.png")
    plt.close()


def validate_TPB_patient_information(filelist: List[str], xlsx_name: str):
    log = open("validate.log", "a", encoding="UTF-8")
    log.write(
        "===> [{}] Validate {} ThreePhaseBone Patients' Information.\n".format(
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"), len(filelist)
        )
    )
    xlsx = pd.read_excel(xlsx_name)
    values = xlsx[["编号", "检查日期", "姓名"]].values

    p = Pinyin()
    for filename in filelist:
        img = pydicom.dcmread(filename)
        img_arr = get_pixel_value(filename)
        index = filename.split("\\")[-1].split("_")[0]
        AcqusitionDate = img.AcquisitionDate
        PatientName = img.PatientName.family_name
        PatientName = PatientName.replace(" ", "")

        xlsx_value = values[int(index) - 1]
        xlsx_date = xlsx_value[1].strip("\t")
        xlsx_date = datetime.strptime(xlsx_date, "%Y-%m-%d").strftime("%Y%m%d")
        xlsx_name = xlsx_value[2].strip("\t")
        xlsx_pinyin = p.get_pinyin(xlsx_name, splitter="", convert="upper")

        if (
            xlsx_pinyin != PatientName
            or AcqusitionDate != xlsx_date
            or img_arr.shape[1:] != (128, 128)
            or (img_arr.shape[0] != 25 and img_arr.shape[0] != 50)
        ):
            log.write(
                "[File Name] {} - [Patient Name] 汉字: {}(in excel), 拼音: {} (in excel), {} (in dicom) - [DateTime] {} (in excel), {} (in dicom) - [IMAGE SHAPE] {}.\n".format(
                    filename,
                    xlsx_name,
                    xlsx_pinyin,
                    PatientName,
                    xlsx_date,
                    AcqusitionDate,
                    img_arr.shape,
                )
            )
            log.flush()
    log.close()


# 校验 PETCT 回归预测数据
reg_data = glob("ProcessedData/regression/new/*.npz")
validate_PETCT_regression_data(reg_data)

# 校验骨三相的病人信息数据
three = glob("ThreePhaseBone/*/*/*FLOW.dcm")
validate_TPB_patient_information(three, "ThreePhaseBone/ThreePhaseBone.xlsx")

# 查看骨三相的注射药剂
log = open("test.log", "w", encoding="UTF-8")
for t in three:
    reader = sitk.ImageFileReader()
    reader.SetFileName(t)
    reader.LoadPrivateTagsOn()
    reader.ReadImageInformation()
    RadioNuclideName = reader.GetMetaData("0011|100d")

    log.write(t + ": " + RadioNuclideName + "\n")
    log.flush()
log.close()
