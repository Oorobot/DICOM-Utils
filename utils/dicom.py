from datetime import datetime
from typing import List, Tuple

import numpy as np
import pydicom
import SimpleITK as sitk


def string_to_datetime(time: str) -> datetime:
    """转换字符串(format: %Y%m%d%H%M%S or %Y%m%d%H%M%S.%f)为 datetime 类型数据"""
    try:
        date_time = datetime.strptime(time, "%Y%m%d%H%M%S")
    except:
        date_time = datetime.strptime(time, "%Y%m%d%H%M%S.%f")
    return date_time


def get_pixel_array(filename: str):
    return pydicom.dcmread(filename).pixel_array


# get_pixel_value(file) = get_pixel_array(file) * RescaleSlope + RescaleIntercept
def get_pixel_value(filename: str):
    return sitk.GetArrayFromImage(sitk.ReadImage(filename))


def read_serises_image(files: List[str]) -> sitk.Image:
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(files)
    images = reader.Execute()
    return images


def convert_dicom_to_gary_image_based_window_center_and_window_width(
    pixel_value: np.ndarray,
    window_center: float,
    window_width: float,
    ratio=1.0,
    to_uint8=False,
):
    if ratio != 1.0:
        window_width *= ratio
        window_center *= ratio

    min_window = window_center - window_width * 0.5
    gray_image = (pixel_value - min_window) / float(window_width)
    np.clip(gray_image, 0, 1, out=gray_image)
    if to_uint8:
        gray_image = (gray_image * 255).astype(np.uint8)
    return gray_image


# CT(DCM格式文件): 获取 Hounsfield Unit 矩阵.
def get_hounsfield_unit(filename: str) -> np.ndarray:
    """根据 CT tag 信息, 计算每个像素值的 hounsfield unit = pixel_array * RescaleSlope + RescaleIntercept"""
    return get_pixel_value(filename)


# PET(DCM格式): 获取 SUVbw 矩阵.
def get_SUVbw(pixel_value: np.ndarray, file: str) -> np.ndarray:
    """来源: https://qibawiki.rsna.org/index.php/Standardized_Uptake_Value_(SUV) 中 "SUV Calculation"
    \n 根据 PET 的 tag 信息, 计算每个像素值的 SUVbw, 不适用于 GE Medical.
    """
    image = pydicom.dcmread(file)

    if (
        "ATTN" in image.CorrectedImage
        and "DECY" in image.CorrectedImage
        and image.DecayCorrection == "START"
    ):
        if image.Units == "BQML":
            half_life = float(
                image.RadiopharmaceuticalInformationSequence[0].RadionuclideHalfLife
            )
            if (
                image.SeriesDate <= image.AcquisitionDate
                and image.SeriesTime <= image.AcquisitionTime
            ):
                scan_datetime = image.SeriesDate + image.SeriesTime
            if (
                "RadiopharmaceuticalStartDateTime"
                in image.RadiopharmaceuticalInformationSequence[0]
            ):
                start_datetime = image.RadiopharmaceuticalInformationSequence[
                    0
                ].RadiopharmaceuticalStartDateTime
            else:
                start_datetime = (
                    image.SeriesDate
                    + image.RadiopharmaceuticalInformationSequence[
                        0
                    ].RadiopharmaceuticalStartTime
                )

            decay_time = (
                string_to_datetime(scan_datetime) - string_to_datetime(start_datetime)
            ).total_seconds()

            injected_dose = float(
                image.RadiopharmaceuticalInformationSequence[0].RadionuclideTotalDose
            )
            decayed_dose = injected_dose * (2 ** (-decay_time / half_life))
            SUVbwScaleFactor = image.PatientWeight * 1000 / decayed_dose
        elif image.Units == "CNTS":
            SUVbwScaleFactor = image.get_item((0x7053, 0x1000))
        elif image.Units == "GML":
            SUVbwScaleFactor = 1.0
    SUVbw = (
        pixel_value * float(image.RescaleSlope) + float(image.RescaleIntercept)
    ) * SUVbwScaleFactor
    return SUVbw


# PET(DCM格式): 获取 SUVbw 矩阵. 使用 GE Medical 的计算方法
def get_SUVbw_in_GE(pixel_value: np.ndarray, file: str) -> np.ndarray:
    """根据 PET 的 tag 信息, 计算每个像素值的 SUVbw. 仅用于 GE medical."""
    image = pydicom.dcmread(file)
    bw = image.PatientWeight * 1000  # g
    decay_time = (
        string_to_datetime(image.SeriesDate + image.SeriesTime)
        - string_to_datetime(
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
    return SUVbw


# PET(DCM格式): 获取 SUVbw, SUVbsa, SUVlbm 矩阵. 使用GE的计算方法
def get_all_SUV_in_GE(
    pixel_value: np.ndarray, file: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """来源: https://qibawiki.rsna.org/images/4/40/Calculation_of_SUVs_in_GE_Apps_v4_%282%29.doc
    根据 PET 的 tag 信息, 计算每个像素值的 SUVbw, SUVbsa, SUVlbm. 仅用于 GE medical."""
    image = pydicom.dcmread(file)

    bw = image.PatientWeight * 1000  # g
    bsa = (
        (image.PatientWeight**0.425)
        * ((image.PatientSize * 100) ** 0.725)
        * 0.007184
        * 10000
    )
    if image.PatientSex == "M":
        lbm = 1.10 * image.PatientWeight - 120 * (
            (image.PatientWeight / (image.PatientSize * 100)) ** 2
        )
    elif image.PatientSex == "F":
        lbm = 1.07 * image.PatientWeight - 148 * (
            (image.PatientWeight / (image.PatientSize * 100)) ** 2
        )

    decay_time = (
        string_to_datetime(image.SeriesDate + image.SeriesTime)
        - string_to_datetime(
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
    SUVbsa = pixel_value * bsa / actual_activity  # cm2/ml
    SUVlbm = pixel_value * lbm * 1000 / actual_activity  # g/ml
    return SUVbw, SUVbsa, SUVlbm
