import os
from typing import List

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


def clip(
    boundary: List[int],  # [left, upper, right, lower]
    cliped_width: int = 64,
    cliped_height: int = 64,
    pic_width: int = 512,
    pic_height: int = 512,
):
    left, upper, right, lower = boundary
    # 这个边界是闭区间，即 [left,right],[upper,lower]，但是随后用在切片中是左闭右开，所以+1
    right += 1
    lower += 1
    need_resize = False
    mask_width = right - left
    mask_height = upper - lower
    center_x = int((left + right) * 0.5)
    center_y = int((upper + lower) * 0.5)

    def get_boundary(center, cliped_length, pic_length):
        assert cliped_length % 2 == 0
        assert pic_length % 2 == 0
        if center - cliped_length * 0.5 <= 0:
            min, max = 0, cliped_width
        elif center + cliped_length * 0.5 >= pic_length:
            min, max = pic_length - cliped_length, pic_length
        else:
            min, max = center - 0.5 * cliped_length, center + 0.5 * cliped_length
        return int(min), int(max)

    if mask_width < cliped_width and mask_height < cliped_height:
        left, right = get_boundary(center_x, cliped_width, pic_width)
        upper, lower = get_boundary(center_y, cliped_height, pic_height)
    elif mask_width >= cliped_width and mask_height < cliped_height:
        upper, lower = get_boundary(center_y, cliped_height, pic_height)
        need_resize = True
    elif mask_width < cliped_width and mask_height >= cliped_height:
        left, right = get_boundary(center_x, cliped_width, pic_width)
        need_resize = True
    else:
        need_resize = True

    return (left, upper, right, lower, need_resize)


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
