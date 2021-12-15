import os
from typing import List, Tuple

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


def clip_based_boundary(
    boundary: Tuple[int, int, int, int],  # [left, upper, right, lower]
    cliped_size: Tuple[int, int] = (64, 64),  # [width, height]
    pic_size: Tuple[int, int] = (512, 512),  # [width, height]
):
    """在一个图片中，以 boundary 为中心剪切出 cliped size 的图片"""

    left, upper, right, lower = boundary
    # boundary 左右均为闭区间, 即[left, right], [upper, lower], 所以 boundary width+1, boundary height+1
    boundary_width = right + 1 - left
    boundary_height = lower + 1 - upper
    # boundary's center
    center_x, center_y = (
        int((left + right) * 0.5),
        int((upper + lower) * 0.5),
    )

    def get_boundary(center, cliped_length, pic_length):
        assert cliped_length % 2 == 0
        assert pic_length % 2 == 0
        if center - cliped_length * 0.5 <= 0:
            min, max = 0, cliped_length
        elif center + cliped_length * 0.5 >= pic_length:
            min, max = pic_length - cliped_length, pic_length
        else:
            min, max = center - 0.5 * cliped_length, center + 0.5 * cliped_length
        return int(min), int(max)

    if boundary_width < cliped_size[0] and boundary_height < cliped_size[1]:
        left, right = get_boundary(center_x, cliped_size[0], pic_size[0])
        upper, lower = get_boundary(center_y, cliped_size[1], pic_size[1])
        out_of_size = False
    elif boundary_width >= cliped_size[0] and boundary_height < cliped_size[1]:
        upper, lower = get_boundary(center_y, cliped_size[1], pic_size[1])
        out_of_size = True
    elif boundary_width < cliped_size[0] and boundary_height >= cliped_size[1]:
        left, right = get_boundary(center_x, cliped_size[0], pic_size[0])
        out_of_size = True
    else:
        out_of_size = True

    # 如果 boundary size 大于 cliped size, out_of_size = True
    return left, upper, right, lower, out_of_size


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


def write_txt(files: List[str], txt: str):
    """将一组文件写入 txt 文件中

    Args:
        files (List[str]): 需要写入txt的一组文件
        txt (str): txt的文件名
    """
    file = open(txt, "w")
    for line in files:
        file.writelines(line + "\n")
    file.close()


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

