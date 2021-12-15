import datetime
import os
import sys
from glob import glob
from typing import List, Tuple

import cv2
import numpy as np
import pydicom
import SimpleITK as sitk


def read_serises_images(files: List[str]) -> sitk.Image:
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(files)
    images = reader.Execute()
    return images


def to_datetime(time: str) -> datetime.datetime:
    """转换字符串(format: %Y%m%d%H%M%S or %Y%m%d%H%M%S.%f)为 datetime 类型数据

    Args:
        time (str): 格式为 %Y%m%d%H%M%S 或 %Y%m%d%H%M%S.%f 的字符串

    Returns:
        datetime.datetime: datetime 类型数据
    """
    try:
        time_ = datetime.datetime.strptime(time, "%Y%m%d%H%M%S")
    except:
        time_ = datetime.datetime.strptime(time, "%Y%m%d%H%M%S.%f")
    return time_


def compute_hounsfield_unit(pixel_value: np.ndarray, file: str) -> np.ndarray:
    """根据 CT tag 信息, 计算每个像素值的 hounsfield unit.

    Args:
        pixel_value (np.ndarray): CT 的像素值矩阵
        file (str): CT 文件相对或绝对路径

    Returns:
        np.ndarray: CT 的 hounsfield unit 矩阵
    """
    image = pydicom.dcmread(file)
    if "RescaleType" in image and image.RescaleType == "HU":
        hounsfield_unit = pixel_value
    else:
        hounsfield_unit = pixel_value * float(image.RescaleSlope) + float(
            image.RescaleIntercept
        )
    # Hounsfield Unit between -1000 and 1000
    np.clip(hounsfield_unit, -1000, 1000, out=hounsfield_unit)
    return hounsfield_unit


def compute_SUVbw_in_GE(pixel_value: np.ndarray, file: str) -> np.ndarray:
    """根据 PET tag 信息, 计算 PET 每个像素值得 SUVbw. 仅用于 GE medical.
    
    Args:
        pixel_value (np.ndarray): PET 的像素值矩阵
        file (str): PET 文件相对或绝对路径
    
    Returns:
        np.ndarray: PET 的 SUVbw 矩阵
    """
    image = pydicom.dcmread(file)
    bw = image.PatientWeight * 1000  # g
    decay_time = (
        to_datetime(image.SeriesDate + image.SeriesTime)
        - to_datetime(
            image.SeriesDate
            + image.RadiopharmaceuticalInformationSequence[
                0
            ].RadiopharmaceuticalStartTime
        )
    ).total_seconds()
    actual_activity = float(
        image.RadiopharmaceuticalInformationSequence[0].RadionuclideTotalDose
    ) * (
        2
        ** (
            -(decay_time)
            / float(
                image.RadiopharmaceuticalInformationSequence[0].RadionuclideHalfLife
            )
        )
    )

    SUVbw = pixel_value * bw / actual_activity  # g/ml

    # SUVbw between 0 and 50
    np.clip(SUVbw, 0, 50, out=SUVbw)
    return SUVbw


def resample(
    img: sitk.Image, tar_img: sitk.Image, is_label: bool = False
) -> sitk.Image:

    resamlper = sitk.ResampleImageFilter()
    resamlper.SetReferenceImage(tar_img)
    resamlper.SetOutputPixelType(sitk.sitkFloat32)
    if is_label:
        resamlper.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        # resamlper.SetInterpolator(sitk.sitkBSpline)
        resamlper.SetInterpolator(sitk.sitkLinear)
    return resamlper.Execute(img)


def get_resampled_SUVbw_from_petct(PET_files: List[str], CT: sitk.Image) -> np.ndarray:
    """对于同一个患者的PETCT, PET将根据CT进行重采样到跟CT一样, 并计算PET的SUVbw.

    Args:
        PET_files (List[str]): 病患的一系列PET切片文件
        CT (sitk.Image): 病患的一系列CT, sitk.Image

    Returns:
        np.ndarray: 与CT一样的三维空间分辨率的suvbw
    """

    pet = read_serises_images(PET_files)
    pet_array = sitk.GetArrayFromImage(pet)

    # 计算每张PET slice的SUVbw
    suvbw = np.zeros_like(pet_array, dtype=np.float32)
    for i in range(pet_array.shape[0]):
        suvbw[i] = compute_SUVbw_in_GE(pet_array[i], PET_files[i])

    # 将suvbw变为图像, 并根据CT进行重采样.
    suvbw_img = sitk.GetImageFromArray(suvbw)
    suvbw_img.SetOrigin(pet.GetOrigin())
    suvbw_img.SetSpacing(pet.GetSpacing())
    suvbw_img.SetDirection(pet.GetDirection())
    resampled_suvbw_img = resample(suvbw_img, CT)

    return sitk.GetArrayFromImage(resampled_suvbw_img)


def reg_data_valid(filelist: List[str]):
    suvmax_max = 0.0
    suvmin_min = 50.0
    abnormal_file = []
    for file in filelist:
        print("file name: ", file)
        data = np.load(file)
        seg = data["seg"]
        hu = data["hu"]
        suvmax = data["suvmax"]
        suvmin = data["suvmin"]
        suvmean = data["suvmean"]

        contours, h = cv2.findContours(
            seg.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE
        )
        print("the number of segmentation: ", len(contours))
        if len(contours) > 1:
            abnormal_file.append(file)
        print(
            "the existence of nan and inf (Ture or False) : ",
            np.isnan(hu).any(),
            np.isinf(hu).any(),
            np.isnan(seg).any(),
            np.isinf(seg).any(),
        )
        print("suv max, min, mean: %f, %f, %f" % (suvmax, suvmin, suvmean))
        if suvmax > suvmax_max:
            suvmax_max = suvmax
        if suvmin < suvmin_min:
            suvmin_min = suvmin
    print("the max of suv max: ", suvmax_max)
    print("the min of suv min: ", suvmin_min)
    print("abnormal files: ", abnormal_file)

