import datetime
import os
from glob import glob
from typing import List, Tuple, Type

import cv2
from matplotlib.pyplot import draw
import numpy as np
import pydicom
import SimpleITK as sitk

from utils import *


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


def resample(img: sitk.Image, tar_img: sitk.Image, is_label: bool) -> sitk.Image:

    resamlper = sitk.ResampleImageFilter()
    resamlper.SetReferenceImage(tar_img)
    resamlper.SetOutputPixelType(sitk.sitkFloat32)
    if is_label:
        resamlper.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        # resamlper.SetInterpolator(sitk.sitkBSpline)
        resamlper.SetInterpolator(sitk.sitkLinear)
    return resamlper.Execute(img)


def only_center_contour(mask: np.ndarray, center: Tuple[float, float]):
    """对具有多个分割区域的mask，消除其余mask，仅保留中心部分的mask"""

    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    if len(contours) != 1:
        draw_list = []
        for idx, contour in enumerate(contours):
            contour = np.squeeze(contour)
            if len(contour.shape) == 1:
                draw_list.append(idx)
                continue
            indices_max = np.max(contour, axis=0)
            indices_min = np.min(contour, axis=0)
            if (
                center[0] < indices_min[1]  # left
                or center[0] > indices_max[1]  # right
                or center[1] < indices_min[0]  # upper
                or center[1] > indices_max[0]  # lower
            ):
                draw_list.append(idx)
        for d in draw_list:
            cv2.drawContours(mask, contours, d, (0, 0, 0), cv2.FILLED)
    return mask


""" series images的顺序是从肺部上面到下面，.nii.gz的顺序恰好相反，从下面到上面
    所以，读取完 .nii.gz 需要将图像序列进行 reverse 
"""

# 常量部分
ALL_MASKS = glob("NewPulmonaryNodule/*/*.nii.gz")
NEW2OLD = np.loadtxt(
    fname="new2lod.csv", dtype=np.uint32, delimiter=",", skiprows=1, usecols=1
)
LUNG_BASE_PATH = "LUNG"
LUNG_SLICE = np.loadtxt(
    fname="lung_slice.csv", dtype=np.uint32, delimiter=",", usecols=(1, 2)
)

if __name__ == "__main__":
    for mask_file in ALL_MASKS:

        print("now start process file: ", mask_file)

        new_dir = mask_file.split("\\")[1]
        old_dir = NEW2OLD[int(new_dir) - 1]

        slice_start, slice_end = LUNG_SLICE[old_dir - 1]

        series_ct_files = glob(
            os.path.join(LUNG_BASE_PATH, str(old_dir).zfill(3), "CT*")
        )
        series_pet_files = glob(
            os.path.join(LUNG_BASE_PATH, str(old_dir).zfill(3), "PET*")
        )
        # 读取CT、PET、mask file
        series_ct = read_serises_images(series_ct_files)
        series_pet = read_serises_images(series_pet_files)
        segmentation = sitk.ReadImage(mask_file)

        ct_array = sitk.GetArrayFromImage(series_ct)
        pet_array = sitk.GetArrayFromImage(series_pet)
        seg_array = sitk.GetArrayFromImage(segmentation)

        # 计算肺部切片长度
        slice_length = slice_end - slice_start + 1

        if slice_length != seg_array.shape[0]:
            print(
                "------the ct'shape is not matched to seg'shape----- the dir is: NewPulmonaryNodule/%s"
                % (new_dir),
            )
            continue

        # 取出CT肺部切片, 文件名由000开始编号，故如此切片
        lung_ct_files = series_ct_files[slice_start : slice_end + 1]
        lung_ct_array = ct_array[slice_start : slice_end + 1]

        # 计算HU
        hu = np.zeros((slice_length, 512, 512), dtype=np.float32)
        for i in range(slice_length):
            hu[i] = compute_hounsfield_unit(lung_ct_array[i], lung_ct_files[i])

        # 由于每张PET的SUVbw与该PET tag info相关，所以依次计算出SUVbw，随后将SUVbw变为图片并重采样到CT一样大小
        suv_bw = np.zeros(pet_array.shape, np.float32)
        for i in range(pet_array.shape[0]):
            suv_bw[i] = compute_SUVbw_in_GE(pet_array[i], series_pet_files[i])

        # 还原suv_bw_img信息
        suv_bw_img = sitk.GetImageFromArray(suv_bw)
        suv_bw_img.SetOrigin(series_pet.GetOrigin())
        suv_bw_img.SetSpacing(series_pet.GetSpacing())
        suv_bw_img.SetDirection(series_pet.GetDirection())

        # 对suv_bw_img重采样
        suv_bw_img = resample(suv_bw_img, series_ct, False)
        suv_bw = sitk.GetArrayFromImage(suv_bw_img)

        # 取出SUV肺部切片, 文件名由000开始编号，故如此切片
        lung_suv_bw = suv_bw[slice_start : slice_end + 1]

        if ct_array.shape != suv_bw.shape or lung_ct_array.shape != lung_suv_bw.shape:
            print("ct and pet is not matched!!!")
            continue

        # seg 顺序与 PET、CT 顺序相反
        for i in range(slice_length):
            """
                保存数据文件格式：npz
                file_name = (old_dir)_(slice_file_name)_(seg_idx).npz
                total: HU(512x512),SUV(512x512),seg(512x512),suvmax,suvmin.suvmean                
                暂时忽略 gen: hu(512x512), suv(512x512), seg(512x512) 
                reg: cliped hu(32x32), cliped seg(32x32), suvmax, suvmin, suvmean
            """

            cur_seg = seg_array[slice_length - i - 1]
            # 有分割图时，进行处理
            if np.max(cur_seg) != 0:
                # 当前获取的数据，CT HU进行归一化
                cur_slice_file_name = lung_ct_files[i].split("\\")[-1][:-4]
                cur_ct_hu = (lung_ct_array[i] + 1000) / 2000.0
                cur_pet_suv = lung_suv_bw[i]
                print(
                    "%s_%s lung slice file is processing!"
                    % (str(old_dir).zfill(3), cur_slice_file_name)
                )

                # 获取掩码后的CT图像
                masked_CT_1 = np.ma.masked_where(cur_seg == 0, cur_ct_hu)
                masked_CT_2 = np.ma.masked_where(cur_seg == 1, cur_ct_hu)
                save_images(
                    [masked_CT_1, masked_CT_2],
                    ["mask 1", "mask 2"],
                    ["bone", "bone"],
                    "process/img/%s_%s_mask.png"
                    % (str(old_dir).zfill(3), cur_slice_file_name),
                )

                # 由于每张图片可能存在多个病灶，所以需要定位出每个病灶并计算出每个病灶的suv max，min，mean
                contours, hierarchy = cv2.findContours(
                    cur_seg.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE
                )

                for idx, contour in enumerate(contours):
                    contour = np.squeeze(contour)
                    if len(contour.shape) == 1:
                        break
                    indices_max = np.max(contour, axis=0)
                    indices_min = np.min(contour, axis=0)
                    # 计算每个病灶的 suv max, suv mean, suv min
                    masked_suv = np.ma.masked_where(cur_seg == 0, cur_pet_suv)
                    cliped_masked_suv = masked_suv[
                        indices_min[1] : indices_max[1] + 1,
                        indices_min[0] : indices_max[0] + 1,
                    ]
                    suv_max = np.max(cliped_masked_suv)
                    suv_min = np.min(cliped_masked_suv)
                    suv_mean = np.mean(cliped_masked_suv)
                    # 在CT中，切出每个病灶
                    clip_rect = clip(
                        [
                            indices_min[1],  # left
                            indices_min[0],  # upper
                            indices_max[1],  # right
                            indices_max[0],  # lower
                        ],
                        cliped_width=32,
                        cliped_height=32,
                    )
                    cliped_image = cur_ct_hu[
                        clip_rect[0] : clip_rect[2], clip_rect[1] : clip_rect[3]
                    ]
                    cliped_seg = cur_seg[
                        clip_rect[0] : clip_rect[2], clip_rect[1] : clip_rect[3]
                    ]
                    if clip_rect[4]:
                        cliped_image = cv2.resize(cliped_image, (32, 32))
                        cliped_seg = cv2.resize(cliped_seg, (32, 32))
                        print("there is one need to resize to 32x32!!")

                    # seg 仅保留一个中心的病灶
                    cliped_seg = only_center_contour(cliped_seg, (15.5, 15.5))

                    # 保存文件
                    save_images(
                        [cliped_image, cliped_seg],
                        ["img", "seg"],
                        ["bone", "gray"],
                        "process/img/%s_%s_%s_cliped.png"
                        % (
                            str(old_dir).zfill(3),
                            cur_slice_file_name,
                            str(idx).zfill(2),
                        ),
                    )
                    np.savez(
                        "process/reg/%s_%s_%s.npz"
                        % (
                            str(old_dir).zfill(3),
                            cur_slice_file_name,
                            str(idx).zfill(2),
                        ),
                        hu=cliped_image,
                        seg=cliped_seg,
                        suvmax=suv_max,
                        suvmean=suv_mean,
                        suvmin=suv_min,
                    )

                    # np.savez(
                    #     "process/gen/%s_%s.npz" % (str(old_dir).zfill(3), str(i).zfill(2)),
                    #     hu=cur_ct_hu,
                    #     suv=cur_pet_suv,
                    #     seg=cur_seg,
                    # )

