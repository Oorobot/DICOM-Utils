from io import TextIOWrapper
import os
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
from utils.utils import OUTPUT_FOLDER, mkdir


def validate_PETCT_regression_data(filelist: List[str], log: TextIOWrapper):
    log.write(
        "\n{} --> Validate {} PET-CT regression data.\n".format(
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
                f"File Name: {filename}, the number of contour: {len(contours)}, [MAX] HU: {np.max(hu)} - SEG: {np.max(seg)}, [MIN] HU: {np.min(hu)} - SEG: {np.min(seg)}.\n"
            )
            log.flush()
    suv = np.reshape(np.array(suv), (-1, 3))
    suv_max = np.max(suv, axis=0)
    suv_min = np.min(suv, axis=0)
    log.write("      SUVmax    SUVmean   SUVmin\n")
    log.write("{:<6}{:<10.4f}{:<10.4f}{:<10.4f}\n".format("max", *suv_max.tolist()))
    log.write("{:<6}{:<10.4f}{:<10.4f}{:<10.4f}\n".format("min", *suv_min.tolist()))

    def bincount(values: np.ndarray):
        num1 = ((0 <= values) & (values < 2.5)).sum()
        num2 = ((2.5 <= values) & (values < 5)).sum()
        num3 = ((5 <= values) & (values < 10)).sum()
        num4 = (10 < values).sum()
        return [num1, num2, num3, num4]

    log.write("SUV   0.0-2.5   2.5-5.0   5.0-10.0  >10.0\n")
    log.write("{:<6}{:<10}{:<10}{:<10}{:<10}\n".format("max", *bincount(suv[:, 0])))
    log.write("{:<6}{:<10}{:<10}{:<10}{:<10}\n".format("mean", *bincount(suv[:, 1])))
    log.write("{:<6}{:<10}{:<10}{:<10}{:<10}\n".format("min", *bincount(suv[:, 2])))
    log.flush()
    log.close()


def validate_TPB_patient_information(
    filelist: List[str], xlsx_name: str, log: TextIOWrapper
):
    log.write(
        "\n{} --> Validate {} ThreePhaseBone Patients' Information.\n".format(
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"), len(filelist)
        )
    )
    xlsx = pd.read_excel(xlsx_name)
    values = xlsx[["编号", "检查日期", "姓名"]].values

    p = Pinyin()
    log.write(
        "{:<20}{:<15}{:<20}{:<20}{:<15}{:<15}{:<12}\n".format(
            "File Name",
            "Patient Name",
            "Excel",
            "DICOM",
            "Datetime_Excel",
            "Datetime_DICOM",
            "Image Shape",
        )
    )
    for file in filelist:
        dicom = pydicom.dcmread(file)
        image = get_pixel_value(file)
        filename = os.path.basename(file)
        index = os.path.basename(os.path.dirname(file))
        AcqusitionDate = dicom.AcquisitionDate
        PatientName = dicom.PatientName.family_name
        PatientName = PatientName.replace(" ", "")

        xlsx_value = values[int(index) - 1]
        xlsx_date = xlsx_value[1].strip("\t")
        xlsx_date = datetime.strptime(xlsx_date, "%Y-%m-%d").strftime("%Y%m%d")
        xlsx_name = xlsx_value[2].strip("\t")
        xlsx_pinyin = p.get_pinyin(xlsx_name, splitter="", convert="upper")

        if (
            xlsx_pinyin != PatientName
            or AcqusitionDate != xlsx_date
            or image.shape[1:] != (128, 128)
            or (image.shape[0] != 25 and image.shape[0] != 50)
        ):
            log.write(
                "{:<20}{:<13}{:<20}{:<20}{:<15}{:<15}{:<12}\n".format(
                    filename,
                    xlsx_name,
                    xlsx_pinyin,
                    PatientName,
                    xlsx_date,
                    AcqusitionDate,
                    str(image.shape),
                )
            )
            log.flush()


# validate 记录文件
mkdir(OUTPUT_FOLDER)
log = open(os.path.join(OUTPUT_FOLDER, "validate.log"), "a", encoding="UTF-8")


# 校验 PETCT 回归预测数据
regresion_data = glob("Data/Files/PETCT/*.npz")
validate_PETCT_regression_data(regresion_data, log)

# 校验骨三相的病人信息数据
# TBs_dicom = glob("Data/ThreePhaseBone/*/*/*FLOW.dcm")
# validate_TPB_patient_information(TBs_dicom, "ThreePhaseBone/ThreePhaseBone.xlsx", log)

# # 查看骨三相的注射药剂
# log = open("test.log", "w", encoding="UTF-8")
# for t in three:
#     reader = sitk.ImageFileReader()
#     reader.SetFileName(t)
#     reader.LoadPrivateTagsOn()
#     reader.ReadImageInformation()
#     RadioNuclideName = reader.GetMetaData("0011|100d")
#     log.write(t + ": " + RadioNuclideName + "\n")
#     log.flush()

log.close()
