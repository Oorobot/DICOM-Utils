import datetime
from typing import List, Tuple

import cv2
import numpy as np
import pydicom
import SimpleITK as sitk


def read_serises_image(files: List[str]) -> sitk.Image:
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


def get_hounsfield_unit(file: str) -> np.ndarray:
    """根据 CT tag 信息, 计算每个像素值的 hounsfield unit.

    Args:
        pixel_value (np.ndarray): CT 的像素值矩阵
        file (str): CT 文件相对或绝对路径

    Returns:
        np.ndarray: CT 的 hounsfield unit 矩阵
    """

    # SimpleITK 读取的 Array = Pydicom 读取的 Array * RescaleSlope + RescaleIntercept
    # return sitk.GetArrayFromImage(sitk.ReadImage(file))
    # 等价与
    image = pydicom.dcmread(file)
    return image.pixel_array * float(image.RescaleSlope) + float(image.RescaleIntercept)


def compute_SUVbw(pixel_value: np.ndarray, file: str) -> np.ndarray:
    """如果特定的 DICOM 文件出现某个属性缺失或为空或为零, SUV将无法计算. GE Medical 不适用该方法."""
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
                to_datetime(scan_datetime) - to_datetime(start_datetime)
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


def compute_SUV_in_GE(
    pixel_value: np.ndarray, file: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """根据 PET tag 信息, 计算 PET 每个像素值的 SUVbw, SUVbsa, SUVlbm. 仅用于 GE medical."""
    image = pydicom.dcmread(file)

    bw = image.PatientWeight * 1000  # g
    bsa = (
        (image.PatientWeight ** 0.425)
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
    SUVbsa = pixel_value * bsa / actual_activity  # cm2/ml
    SUVlbm = pixel_value * lbm * 1000 / actual_activity  # g/ml
    return SUVbw, SUVbsa, SUVlbm


def compute_SUVbw_in_GE(pixel_value: np.ndarray, file: str) -> np.ndarray:
    """根据 PET tag 信息, 计算 PET 每个像素值的 SUVbw. 仅用于 GE medical.
    
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


def get_resampled_SUVbw_from_PETCT(
    PET_files: List[str], CT_image: sitk.Image
) -> np.ndarray:
    """对于同一个患者的PETCT, PET将根据CT进行重采样到跟CT一样, 并计算PET的SUVbw.

    Args:
        PET_files (List[str]): 病患的一系列PET切片文件
        CT (sitk.Image): 病患的一系列CT, sitk.Image

    Returns:
        np.ndarray: 与CT一样的三维空间分辨率的suvbw
    """

    PET_image = read_serises_image(PET_files)
    PET_array = sitk.GetArrayFromImage(PET_image)

    # 计算每张PET slice的SUVbw
    SUVbw = np.zeros_like(PET_array, dtype=np.float32)
    for i in range(PET_array.shape[0]):
        SUVbw[i] = compute_SUVbw_in_GE(PET_array[i], PET_files[i])

    # 将suvbw变为图像, 并根据CT进行重采样.
    SUVbw_image = sitk.GetImageFromArray(SUVbw)
    SUVbw_image.SetOrigin(PET_image.GetOrigin())
    SUVbw_image.SetSpacing(PET_image.GetSpacing())
    SUVbw_image.SetDirection(PET_image.GetDirection())
    resampled_SUVbw_image = resample(SUVbw_image, CT_image)

    return sitk.GetArrayFromImage(resampled_SUVbw_image)


def regression_data_validate(filelist: List[str], txt: str):
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
