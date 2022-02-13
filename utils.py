import json
import os
from tkinter.tix import Tree
from typing import List, Tuple
import datetime
import stat

import cv2
import matplotlib.pyplot as plt
import numpy as np


def save_image(img, cmap, path):
    plt.axis("off")
    plt.imshow(img, cmap=cmap)
    plt.savefig(path)
    plt.close()


def save_images(imgs, titles, camps, path):
    for i in range(len(imgs)):
        plt.subplot(1, len(imgs), i + 1)
        plt.title(titles[i])
        plt.imshow(imgs[i], camps[i])
        plt.axis("off")
    plt.savefig(path)
    plt.close()


def mkdir(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)


def mkdirs(dirs):
    if isinstance(dirs, list) and not isinstance(dirs, str):
        for dir in dirs:
            mkdir(dir)
    else:
        mkdir(dirs)


def rename(src, dst):
    try:
        os.rename(src, dst)
    except (FileNotFoundError):
        print("the dir is not existed.")


def delete(filename):
    try:
        os.remove(filename)
    except:
        print("the file is unable to delete directly.")
        os.chmod(filename, stat.S_IWRITE)
        os.remove(filename)


def save_json(save_path: str, data: dict):
    assert save_path.split(".")[-1] == "json"
    with open(save_path, "w") as file:
        json.dump(data, file)


def load_json(file_path: str):
    assert file_path.split(".")[-1] == "json"
    with open(file_path, "r") as file:
        data = json.load(file)
    return data


def string_to_datetime(time: str) -> datetime.datetime:
    """转换字符串(format: %Y%m%d%H%M%S or %Y%m%d%H%M%S.%f)为 datetime 类型数据"""
    try:
        date_time = datetime.datetime.strptime(time, "%Y%m%d%H%M%S")
    except:
        date_time = datetime.datetime.strptime(time, "%Y%m%d%H%M%S.%f")
    return date_time


def files_split(files: List[str], ratio: float):
    """将一组文件划分为两组文件, 该两组文件数量之比为 ratio.

    Args:
        files (List[str]): 需要划分的一组文件
        ratio (float): 划分的比例

    Returns:
        划分后的两组文件
    """
    num = int(ratio * len(files))
    selected_files = np.random.choice(files, num, False)
    left_files = np.setdiff1d(files, selected_files)
    return selected_files, left_files


def crop_based_lesions(
    boundary: Tuple[int, int, int, int],  # [left, upper, right, lower]
    cliped_size: Tuple[int, int] = (32, 32),  # [width, height]
    pic_size: Tuple[int, int] = (512, 512),  # [width, height]
):
    """以病灶为中心剪切出 cliped size 的图片"""

    left, upper, right, lower = boundary
    # boundary 左右均为闭区间, 即[left, right], [upper, lower].
    boundary_width = right - left
    boundary_height = lower - upper
    # boundary's center
    center_x, center_y = (
        (left + right) * 0.5,
        (upper + lower) * 0.5,
    )

    def get_boundary(center, cliped_length, pic_length):
        if center - cliped_length * 0.5 <= 0:
            min, max = 0, cliped_length + 1
        elif center + cliped_length * 0.5 >= pic_length:
            min, max = pic_length - cliped_length - 1, pic_length
        else:
            min, max = center - 0.5 * cliped_length, center + 0.5 * cliped_length + 1
        return int(min), int(max)

    if boundary_width < cliped_size[0] and boundary_height < cliped_size[1]:
        left, right = get_boundary(center_x, cliped_size[0] - 1, pic_size[0])
        upper, lower = get_boundary(center_y, cliped_size[1] - 1, pic_size[1])
        apply_resize = False
    # 基于病灶中心按照病灶的边界的最长边进行切片(正方形), 随后resize为32x32
    else:
        max_length = max(boundary_height, boundary_width)
        left, right = get_boundary(center_x, max_length, pic_size[0])
        upper, lower = get_boundary(center_y, max_length, pic_size[1])
        apply_resize = True
    return left, upper, right, lower, apply_resize


def only_center_contour(mask: np.ndarray, center: Tuple[float, float]):
    """在图像中, 对具有多个分割区域的mask, 消除其余mask, 仅保留中心部分的mask."""

    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)

    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    if len(contours) != 1:
        draw_list = []
        for idx, contour in enumerate(contours):
            contour = np.squeeze(contour)
            if len(contour.shape) == 1:
                draw_list.append(idx)
                continue
            contour_right, contour_lower = np.max(contour, axis=0)
            contour_left, contour_upper = np.min(contour, axis=0)
            if (
                center[0] < contour_left
                or center[0] > contour_right
                or center[1] < contour_upper
                or center[1] > contour_lower
            ):
                draw_list.append(idx)
        for d in draw_list:
            cv2.drawContours(mask, contours, d, (0, 0, 0), cv2.FILLED)
    return mask


# pic = np.zeros((512, 512))
# left, upper, right, lower, apply_resize = crop_based_boundary([20, 20, 60, 70])
# pic1 = pic[upper:lower, left:right]
# left, upper, right, lower, apply_resize = crop_based_boundary([20, 20, 59, 69])
# pic2 = pic[upper:lower, left:right]

# print(0)
