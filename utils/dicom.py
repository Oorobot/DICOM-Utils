import math
from typing import List, Tuple

import cv2
import numpy as np
import pydicom
import SimpleITK as sitk

from utils.utils import str2datetime

# -----------------------------------------------------------#
#                     读取一系列的图像
# -----------------------------------------------------------#
def read_serises_image(files: List[str]) -> sitk.Image:
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(files)
    images = reader.Execute()
    return images


def read_serises_SUVbw_image(pet_files: List[str]) -> sitk.Image:
    pet_image = read_serises_image(pet_files)
    pet_array = sitk.GetArrayFromImage(pet_image)
    SUVbw_array = np.zeros_like(pet_array)
    for i in range(pet_array.shape[0]):
        SUVbw_array[i] = get_SUV_in_GE(pet_array[i], pet_files[i])[0]
    SUVbw_image = sitk.GetImageFromArray(SUVbw_array)
    SUVbw_image.CopyInformation(pet_image)
    return SUVbw_image


# --------------------------------------------------------------------------------#
#                             读取Dicom文件的矩阵数据
# get_pixel_value(file) = get_pixel_array(file) * RescaleSlope + RescaleIntercept #
# --------------------------------------------------------------------------------#
def get_pixel_array(filename: str):
    return pydicom.dcmread(filename).pixel_array


def get_pixel_value(filename: str):
    return sitk.GetArrayFromImage(sitk.ReadImage(filename))


# -----------------------------------------------------------#
#           计算单张 CT(Dicom) 的 Hounsfield Unit
# -----------------------------------------------------------#
def get_hounsfield_unit(filename: str) -> np.ndarray:
    """hounsfield unit = pixel_array * RescaleSlope + RescaleIntercept"""
    return get_pixel_value(filename)


# -----------------------------------------------------------#
#               计算单张 PET(Dicom) 的 SUVbw
# -----------------------------------------------------------#
def get_SUVbw(pixel_value: np.ndarray, file: str) -> np.ndarray:
    """来源: https://qibawiki.rsna.org/index.php/Standardized_Uptake_Value_(SUV) 中 "SUV Calculation"。
    \n 不适用于来源于 GE Medical 的 Dicom 文件。
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
                str2datetime(scan_datetime) - str2datetime(start_datetime)
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


# -----------------------------------------------------------#
#        计算来源于 GE Medical 的单张 PET(Dicom) 的 SUV
# -----------------------------------------------------------#
def get_SUV_in_GE(
    pixel_value: np.ndarray, file: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """来源: https://qibawiki.rsna.org/images/4/40/Calculation_of_SUVs_in_GE_Apps_v4_%282%29.doc。
    \n 仅适用于 GE medical
    \n 返回 SUVbw, SUVbsa, SUVlbm
    """
    image = pydicom.dcmread(file)
    try:
        bw = image.PatientWeight * 1000  # g
        bsa = (
            (image.PatientWeight**0.425)
            * ((image.PatientSize * 100) ** 0.725)
            * 0.007184
            * 10000
        )
        lbm = (
            (
                1.10 * image.PatientWeight
                - 120 * ((image.PatientWeight / (image.PatientSize * 100)) ** 2)
            )
            if image.PatientSex == "M"  # 性别为男
            else (  # 性别为女
                1.07 * image.PatientWeight
                - 148 * ((image.PatientWeight / (image.PatientSize * 100)) ** 2)
            )
        )
        decay_time = (
            str2datetime(image.SeriesDate + image.SeriesTime)
            - str2datetime(
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
    except:
        print("文件存在信息缺失")

    SUVbw = pixel_value * bw / actual_activity  # g/ml
    SUVbsa = pixel_value * bsa / actual_activity  # cm2/ml
    SUVlbm = pixel_value * lbm * 1000 / actual_activity  # g/ml
    return SUVbw, SUVbsa, SUVlbm


# -----------------------------------------------------------#
#                     获取图像的信息
# -----------------------------------------------------------#
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


# -----------------------------------------------------------#
#                      图像转换
# -----------------------------------------------------------#
def HU2image(
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


def SUVbw2image(pixel_value: np.ndarray, SUVbw_max: float, to_uint8: bool = False):
    np.clip(pixel_value, 0, SUVbw_max, pixel_value)
    image = pixel_value / SUVbw_max
    if to_uint8:
        image = (image * 255).astype(np.uint8)
    return image


# -----------------------------------------------------------#
#                           重采样
# -----------------------------------------------------------#
def resample_based_target_image(
    image: sitk.Image, target_image: sitk.Image, is_label: bool = False
) -> sitk.Image:
    """
    将数据重采样的指定的图像一致
    image: sitk 读取的 image 数据
    target_image: 指定的 image
    return: 重采样后的 image
    """
    outputPixelType = sitk.sitkInt32 if is_label else sitk.sitkFloat32
    interpolator = sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear
    return sitk.Resample(
        image,
        target_image,
        interpolator=interpolator,
        outputPixelType=outputPixelType,
        useNearestNeighborExtrapolator=is_label,
    )


def resample_based_spacing(
    image: sitk.Image, output_spacing: list, is_label: bool = False
):
    """
    将数据重采样的指定的 spacing 大小
    image: sitk 读取的 image 据
    output_spacing: 指定的 spacing
    return: 重采样后的 image
    """
    input_size = image.GetSize()
    input_spacing = image.GetSpacing()
    output_size = [
        math.ceil(input_size[i] * input_spacing[i] / output_spacing[i])
        for i in range(3)
    ]
    interpolator = sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear
    return sitk.Resample(
        image,
        size=output_size,
        interpolator=interpolator,
        outputOrigin=image.GetOrigin(),
        outputSpacing=output_spacing,
        outputPixelType=image.GetPixelID(),
        outputDirection=image.GetDirection(),
    )


def resameple_based_size(image: sitk.Image, output_size: list, is_label: bool = False):
    """
    将数据重采样的指定的 size
    image: sitk 读取的 image 数据
    output_size: 指定的 size
    return: 重采样后的 image
    """
    input_size = image.GetSize()
    input_spacing = image.GetSpacing()
    output_spacing = [
        input_size[i] * input_spacing[i] / output_size[i] for i in range(3)
    ]
    interpolator = sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear
    return sitk.Resample(
        image,
        size=output_size,
        interpolator=interpolator,
        outputOrigin=image.GetOrigin(),
        outputSpacing=output_spacing,
        outputPixelType=image.GetPixelID(),
        outputDirection=image.GetDirection(),
    )


# -----------------------------------------------------------#
#                   获取三维标注（立方体）
# -----------------------------------------------------------#
def get_3D_annotation(label_path: str) -> Tuple[List[int], List[List[int]]]:
    """ 
    读取itk-snap软件标注的三维边界框以及类别信息
    label_path: 标注数据文件(.nii.gz)路径
    return: 类别 [c1, ...] 和 三维边界框 [(x1, y1, z1, x2, y2, z2), ...]\n
    3D Bounding Box
            •------------------•
           /¦                 /¦
          / ¦                / ¦
         /  ¦               /  ¦
        •------------------V2  ¦
        ¦   ¦              ¦   ¦
        ¦   ¦              ¦   ¦
        ¦   ¦              ¦   ¦
        ¦   ¦              ¦   ¦
        ¦   ¦              ¦   ¦           z
        ¦   V1-------------¦---•           ⋮
        ¦  /               ¦  /            o ⋯ x
        ¦ /                ¦ /           ⋰
        ¦/                 ¦/           y
        •------------------• 
    """
    label_array = get_pixel_value(label_path)
    num_category = int(np.max(label_array))  # 类的数量
    categories = []  # 类别
    annotations = []  # 标注
    for category in range(1, num_category + 1):  # 遍历所有类
        category_array = np.where(label_array == category, 1, 0)
        for i, slice in enumerate(category_array):  # 遍历所有切片
            if 0 == np.max(slice):  # 不存在标注 --> 跳过
                continue
            xy_contours, _ = cv2.findContours(
                slice.astype(np.uint8),
                cv2.RETR_LIST,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            for xy_contour in xy_contours:  # 遍历 xy 平面的标注
                xy_contour = np.squeeze(xy_contour, axis=1)
                assert (
                    len(xy_contour) == 4
                ), f"({xy_contour[0,0]}, {xy_contour[0,1]}, {i}) 处存在不规整的标注"
                x1, y1 = xy_contour[0]
                x2, y2 = xy_contour[2]

                # 根据 x1 找到相应 yz 平面的标注
                yz_contours, _ = cv2.findContours(
                    category_array[:, :, x1].astype(np.uint8),
                    cv2.RETR_LIST,
                    cv2.CHAIN_APPROX_SIMPLE,
                )
                for yz_contour in yz_contours:  # 遍历 yz 平面的标注
                    yz_contour = np.squeeze(yz_contour, axis=1)
                    assert (
                        len(yz_contour) == 4
                    ), f"({x1}, {yz_contour[0,0]}, {yz_contour[0,1]}) 处存在不规整的标注"
                    y1_, z1 = yz_contour[0]
                    y2_, z2 = yz_contour[2]

                    if y1 == y1_ and y2 == y2_:
                        categories.append(category)
                        annotations.append(
                            [int(x1), int(y1), int(z1), int(x2), int(y2), int(z2)]
                        )
                        # 清除对应的立方体标注
                        category_array[z1 : z2 + 1, y1 : y2 + 1, x1 : x2 + 1] = 0
                        break
    return categories, annotations


# -----------------------------------------------------------#
#                      形态学分割方法
# -----------------------------------------------------------#
def get_max_component(mask_image: sitk.Image) -> sitk.Image:
    # 得到mask中的多个连通量
    cc_filter = sitk.ConnectedComponentImageFilter()
    cc_filter.SetFullyConnected(True)

    output_image = cc_filter.Execute(mask_image)

    # 计算不同连通图的大小
    lss_filter = sitk.LabelShapeStatisticsImageFilter()
    lss_filter.Execute(output_image)

    label_num = cc_filter.GetObjectCount()
    max_label = 0
    max_num = 0

    for i in range(1, label_num + 1):
        num = lss_filter.GetNumberOfPixels(i)
        if num > max_num:
            max_label = i
            max_num = num
    return sitk.Equal(output_image, max_label)


def get_binary_image(image: sitk.Image, threshold: int = -200) -> sitk.Image:
    """
    将图像进行二值化: CT threshold = -200, SUVbw threshold = 3e-2
    """
    return sitk.BinaryThreshold(image, lowerThreshold=threshold, upperThreshold=1e8)


def binary_morphological_closing(mask_image: sitk.Image, kernel_radius=2)->sitk.Image:
    bmc_filter = sitk.BinaryMorphologicalClosingImageFilter()
    bmc_filter.SetKernelType(sitk.sitkBall)
    bmc_filter.SetKernelRadius(kernel_radius)
    bmc_filter.SetForegroundValue(1)
    return bmc_filter.Execute(mask_image)


def binary_morphological_opening(mask_image: sitk.Image, kernel_radius=2):
    bmo_filter = sitk.BinaryMorphologicalOpeningImageFilter()
    bmo_filter.SetKernelType(sitk.sitkBall)
    bmo_filter.SetKernelRadius(kernel_radius)
    bmo_filter.SetForegroundValue(1)
    return bmo_filter.Execute(mask_image)


def get_body(
    ct_image: sitk.Image,
    suv_image: sitk.Image,
):
    # 忽略其中进行闭操作和最大连通量
    # CT(去除机床) = CT binary ∩ SUV binary; CT(去除伪影) = CT binary(去除机床) ∪ SUV binary;
    # CT(去除伪影) ≈ SUV binary
    ct_binary = get_binary_image(ct_image, -200)
    suv_binary = get_binary_image(suv_image, 3e-2)

    # 对CT进行闭操作，取最大连通量
    ct_binary_closing_max = get_max_component(
        binary_morphological_closing(ct_binary)
    )
    # 对SUV进行闭操作，取最大连通量
    suv_binary_closing_max = get_max_component(
        binary_morphological_closing(suv_binary)
    )

    ct_no_machine = sitk.And(ct_binary_closing_max, suv_binary_closing_max)
    # 取最大连通量
    ct_no_machine_max = get_max_component(ct_no_machine)

    # 使用超大半径的闭操作，消除伪影
    ct_no_machine_max_closing = binary_morphological_closing(ct_no_machine_max, 20)
    return ct_no_machine_max_closing
 