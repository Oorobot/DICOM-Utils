from datetime import datetime
from typing import List

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydicom
import SimpleITK as sitk
from xpinyin import Pinyin


# 处理骨三相
def img_process(filename: str, crop_type: str, label: int, save_path: str):

    if crop_type == "3 4":
        x = 130
        y = 77
        pic_width = 160
        pic_height = 120
        displacement_x = 406
        displacement_y = 169
    elif crop_type == "4 4":
        x = 155
        y = 86 + 125
        pic_width = 110
        pic_height = 86
        displacement_x = 405
        displacement_y = 125
    else:
        return

    # 读取图像(灰度图)
    image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    # 裁剪下来的16张图象, blood flow phase 前12张, blood pool phase 后4张
    cropped_images = []
    # 每张裁剪下来后resize到256x256的图像
    resized_images = []

    for i in range(3):
        for j in range(4):
            an_image = image[
                y + i * displacement_y : y + pic_height + i * displacement_y,
                x + j * displacement_x : x + pic_width + j * displacement_x,
            ]
            cropped_images.append(an_image)
            # resize -> 256x256
            resized_images.append(cv2.resize(an_image, (256, 256)))

    x = 294
    y = 602
    pic_width = 240
    pic_height = 180
    displacement_x = 813
    displacement_y = 253

    for i in range(2):
        for j in range(2):
            an_image = image[
                y + i * displacement_y : y + pic_height + i * displacement_y,
                x + j * displacement_x : x + pic_width + j * displacement_x,
            ]
            # resize -> 256x256
            cropped_images.append(an_image)
            # resize -> 256x256
            resized_images.append(cv2.resize(an_image, (256, 256)))

    resized_images = np.array(resized_images)
    # 保存裁剪的图片
    for i in range(16):
        plt.subplot(4, 4, i + 1)
        plt.imshow(cropped_images[i], plt.cm.gray)
        plt.axis("off")
    plt.savefig(save_path + "_JPG_CROP.png")
    plt.close()
    # 保存resize后的图片
    for i in range(16):
        plt.subplot(4, 4, i + 1)
        plt.imshow(resized_images[i], plt.cm.gray)
        plt.axis("off")
    plt.savefig(save_path + "_JPG_RESIZE.png")
    plt.close()
    # 保存数据文件
    np.savez(save_path + "_JPG.npz", data=resized_images, label=label)


def get_dcm_array(filename: str):
    img = sitk.ReadImage(filename)
    img_array = sitk.GetArrayFromImage(img)
    return img_array


def dcm_process(filename: str, label: int, save_path: str):
    raw_images = get_dcm_array(filename)
    images = raw_images[0:25]
    # 保存图片
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.imshow(images[i], plt.cm.binary)
        plt.axis("off")
    plt.savefig(save_path + "_DCM.png")
    plt.close()
    # 保存数据文件
    np.savez(save_path + "_DCM.npz", data=images, label=label)


# 校验dcm数据是否匹配xlsx中的数据
def validate_dcm(flows: List[str], xlsx_name: str):
    xlsx = pd.read_excel(xlsx_name, sheet_name="Sheet1", usecols=[0, 1, 2])

    p = Pinyin()
    values = xlsx.values
    for flow in flows:
        img = pydicom.dcmread(flow)
        img_arr = get_dcm_array(flow)
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

