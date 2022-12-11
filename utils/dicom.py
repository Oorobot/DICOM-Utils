import math
from datetime import datetime
from typing import List, Tuple

import cv2
import numpy as np
import pydicom
import SimpleITK as sitk


def string2datetime(time: str) -> datetime:
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
                string2datetime(scan_datetime) - string2datetime(start_datetime)
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
        string2datetime(image.SeriesDate + image.SeriesTime)
        - string2datetime(
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
        string2datetime(image.SeriesDate + image.SeriesTime)
        - string2datetime(
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


DICOM_TAG = {
    0x00100010: "Patient Name",
    0x00100020: "Patient ID",
    0x00100040: "Patient Sex",
    0x00080022: "Acquisition Date",
    0x00080080: "Institution Name",
    0x00080081: "Institution Adresss",
    0x00080070: "Manufacturer",
    0x00081010: "Station Name",
    0x00081090: "Manufacturer Model Name",
}


def read_serises_image(files: List[str]) -> sitk.Image:
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(files)
    images = reader.Execute()
    return images


def get_patient_info(filename: str, dicom_tag: dict = DICOM_TAG):
    file = pydicom.dcmread(filename)
    information = {}
    for key, value in dicom_tag.items():
        try:
            information[value] = file[key].value
        except:
            print(f"{value} is not exist")
            information[value] = None
    return information


def ct2image(
    pixel_value: np.ndarray,
    window_center: float,
    window_width: float,
    to_uint8: bool = False,
):
    window_min = window_center - window_width * 0.5
    window_max = window_center + window_width * 0.5
    np.clip(pixel_value, window_min, window_max, pixel_value)
    image = (pixel_value - window_min) / window_width
    if to_uint8:
        image = (image * 255).astype(np.uint8)
    return image


def suvbw2image(pixel_value: np.ndarray, suvbw_max: float, to_uint8: bool = False):
    np.clip(pixel_value, 0, suvbw_max, pixel_value)
    image = pixel_value / suvbw_max
    if to_uint8:
        image = (image * 255).astype(np.uint8)
    return image


def resample_based_target_image(
    image: sitk.Image, target_image: sitk.Image, is_label: bool = False
) -> sitk.Image:
    outputPixelType = sitk.sitkInt64 if is_label else sitk.sitkFloat32
    return sitk.Resample(
        image,
        target_image,
        outputPixelType=outputPixelType,
        useNearestNeighborExtrapolator=is_label,
    )


def resample_based_spacing(
    image: sitk.Image, output_spacing: list, is_label: bool = False
):
    """
    将体数据重采样的指定的 spacing 大小
    image: sitk 读取的 image 信息, 这里是体数据
    output_spacing: 指定的 spacing
    return: 重采样后的数据
    """

    # 读取文件的size和spacing信息
    input_size = image.GetSize()
    input_spacing = image.GetSpacing()
    # 计算改变spacing后的size，用物理尺寸/体素的大小
    output_size = [
        math.ceil(input_size[i] * input_spacing[i] / output_spacing[i])
        for i in range(3)
    ]
    return sitk.Resample(
        image,
        size=output_size,
        outputOrigin=image.GetOrigin(),
        outputSpacing=output_spacing,
        outputPixelType=image.GetPixelID(),
        outputDirection=image.GetDirection(),
        useNearestNeighborExtrapolator=is_label,
    )


def resameple_based_size(image: sitk.Image, output_size: list, is_label: bool = False):
    """
    将体数据重采样的指定的 size
    image: sitk 读取的 image 信息, 这里是体数据
    output_size: 指定的 size
    return: 重采样后的数据
    """
    # 读取文件的size
    input_size = image.GetSize()
    input_spacing = image.GetSpacing()
    output_spacing = [
        input_size[i] * input_spacing[i] / output_size[i] for i in range(3)
    ]
    return sitk.Resample(
        image,
        size=output_size,
        outputOrigin=image.GetOrigin(),
        outputSpacing=output_spacing,
        outputPixelType=image.GetPixelID(),
        outputDirection=image.GetDirection(),
        useNearestNeighborExtrapolator=is_label,
    )


def get_3D_annotation(label: np.ndarray):
    # 获取标注数据的类数量
    class_number = int(np.max(label))
    # 寻找每个类的标注
    classes = []
    annotations = []
    for class_value in range(1, class_number + 1):
        # 选择的类为 1, 其余为 0.
        class_label_array = np.where(label != class_value, 0, 1)
        for i, slice in enumerate(class_label_array):
            # 切片中不存在标注
            if 0 == np.max(slice):
                continue
            # 存在标注
            xy_contours, _ = cv2.findContours(
                slice.astype(np.uint8),
                cv2.RETR_LIST,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            # 遍历 xy 平面的标注
            for xy_contour in xy_contours:
                xy_contour = np.squeeze(xy_contour, axis=1)
                assert (
                    len(xy_contour) == 4
                ), f"({xy_contour[0,0]}, {xy_contour[0,1]}, {i}) 处存在不规整的标注"
                x1, y1 = xy_contour[0]
                x2, y2 = xy_contour[2]
                # 根据 x1 找到相应 yz 平面的标注
                yz_contours, _ = cv2.findContours(
                    class_label_array[:, :, x1].astype(np.uint8),
                    cv2.RETR_LIST,
                    cv2.CHAIN_APPROX_SIMPLE,
                )
                # 遍历 yz 平面的标注
                for yz_contour in yz_contours:
                    assert (
                        len(yz_contour) == 4
                    ), f"({x1}, {yz_contour[0,0]}, {yz_contour[0,1]}) 处存在不规整的标注"
                    yz_contour = np.squeeze(yz_contour)
                    y1_, z1 = yz_contour[0]
                    y2_, z2 = yz_contour[2]
                    if y1 == y1_ and y2 == y2_:
                        classes.append(class_value)
                        annotations.append(
                            [int(x1), int(y1), int(z1), int(x2), int(y2), int(z2)]
                        )
                        # 清除对应的立方体标注
                        class_label_array[z1 : z2 + 1, y1 : y2 + 1, x1 : x2 + 1] = 0
    return classes, annotations
