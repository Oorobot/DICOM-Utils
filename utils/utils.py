import json
import os
import stat
from datetime import datetime
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from xpinyin import Pinyin

# -----------------------------------------------------------#
#                            常量
# -----------------------------------------------------------#
BASE_FOLDER = "./Files/FRI/image_2mm"
OUTPUT_FOLDER = "./Files"
COLORS = ["#63b2ee", "#76da91", "#f8cb7f"]
LABELS = json.load(open("./Files/FRI/image_2mm.json"))
P = Pinyin()


# -----------------------------------------------------------#
#                        汉字 --> 拼音
# -----------------------------------------------------------#
def to_pinyin(chinese_characters: str):
    return P.get_pinyin(chinese_characters.strip(), " ", convert="upper")


# -----------------------------------------------------------#
#                          文件处理
# -----------------------------------------------------------#
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
    except FileNotFoundError:
        print("the dir is not existed.")


def delete(filename):
    try:
        os.remove(filename)
    except:
        print("the file is unable to delete directly.")
        os.chmod(filename, stat.S_IWRITE)
        os.remove(filename)


def is_empty(dir: str):
    return not os.listdir(str)


# -----------------------------------------------------------#
#                            Json
# -----------------------------------------------------------#
def save_json(save_path: str, data: dict):
    assert save_path.split(".")[-1] == "json"
    with open(save_path, "w") as file:
        json.dump(data, file)


def load_json(file_path: str):
    assert file_path.split(".")[-1] == "json"
    with open(file_path, "r") as file:
        data = json.load(file)
    return data


# -----------------------------------------------------------#
#                           直方图
# -----------------------------------------------------------#
def plot_mutilhist(
    a: List[list],
    bins: List[list],
    colors: List[str],
    labels: List[str],
    xlabel: str,
    ylabel: str,
):
    assert len(a) == len(colors) == len(labels), "参数的类型长度不匹配"
    n = len(a)  # n个一维直方图
    width = 0.95 if n == 1 else 1.0 / n  # 设置每个直方的宽度
    plt.figure(figsize=(19.2, 10.8), dpi=100)
    # 计算不同类型下在bins中各自的数量
    height = [np.histogram(_, bins)[0] for _ in a]
    left = np.arange(len(bins) - 1)
    ax = plt.subplot(111)
    for i, h in enumerate(height):
        bar = ax.bar(
            left + (i + 0.5) / n, h, width=width, color=colors[i], label=labels[i]
        )
        ax.bar_label(bar)
    ax.legend()
    ax.set_xticks(np.arange(0, len(bins)))
    ax.set_xticklabels(map(str, bins))
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    plt.show()


# -----------------------------------------------------------#
#                     字符串 --> datatime
# -----------------------------------------------------------#
def str2datetime(time: str) -> datetime:
    """转换字符串(format: %Y%m%d%H%M%S or %Y%m%d%H%M%S.%f)为 datetime 类型数据"""
    date_time = None
    try:
        date_time = datetime.strptime(time, "%Y%m%d%H%M%S")
    except:
        print("无法转换为'%Y%m%d%H%M%S'形式。")
    try:
        date_time = datetime.strptime(time, "%Y%m%d%H%M%S.%f")
    except:
        print("无法转换为'%Y%m%d%H%M%S.%f'形式。")
    return date_time


# ---------------------------------------------------#
#               三维医学影像的数据增强
# ---------------------------------------------------#
def random_flip(image: np.ndarray, box: np.ndarray, axis=None):
    # 选择翻转的轴 x, y, z
    axis = np.random.randint(0, 3) if axis is None else axis
    # image's shape [C, D, H, W]
    image = np.flip(image, 3 - axis)
    axis_size = image.shape[3 - axis]
    # b's shape [x1, y1, z1, x2, y2, z2, c]
    for b in box:
        b[axis], b[axis + 3] = axis_size - b[axis + 3], axis_size - b[axis]
    return image, box


def random_rot90(ct_patch: np.ndarray, suv_patch: np.ndarray, boxes: list):
    k = np.random.randint(0, 4)
    _axes = [(0, 1), (0, 2), (1, 2)]
    axes = _axes[np.random.randint(0, 3)]
    ct_p = np.rot90(ct_patch, k, axes)
    suv_p = np.rot90(suv_patch, k, axes)
    b = []
    for box in boxes:
        # box: [x1, y1, z1, x2, y2, z2, c]
        point1 = list(reversed(box[0:3]))
        point2 = list(reversed(box[3:6]))
        point1, point2 = rot90_3D_annotation(ct_patch.shape, point1, point2, k, axes)
        b.append(list(reversed(point1)) + list(reversed(point2)) + [box[-1]])
    return ct_p, suv_p, b


def rot90_3D_annotation(image_size, point1, point2, k, axes):
    """
    image_size = (D, H, W): 三维物体的大小
    point1 = (z1, y1, x1): 三维标注中对角的第一个点
    point2 = (z2, y2, x2): 三维标注中对角的第二个点(x2>x1, y2>y1, z2>z1)
    k: 旋转90度的次数
    axes: 按照哪个平面进行旋转, 在x-y平面, axes = (1, 2) 或者 axes = (2, 1)
    """
    k = k % 4
    y1, x1 = point1[axes[0]], point1[axes[1]]
    y2, x2 = point2[axes[0]], point2[axes[1]]
    H, W = image_size[axes[0]], image_size[axes[1]]
    if k == 0:
        return point1, point2
    elif k == 1:
        x1_, y1_ = y1, W - x2
        x2_, y2_ = y2, W - x1
    elif k == 2:
        x1_, y1_ = W - x2, H - y2
        x2_, y2_ = W - x1, H - y1
    elif k == 3:
        x1_, y1_ = H - y2, x1
        x2_, y2_ = H - y1, x2
    point1[axes[0]], point1[axes[1]] = y1_, x1_
    point2[axes[0]], point2[axes[1]] = y2_, x2_
    return point1, point2


def mixup(image1: np.ndarray, box1: np.ndarray, image2: np.ndarray, box2: np.ndarray):
    image = image1 * 0.5 + image2 * 0.5
    if len(box1) == 0:
        box = box2
    elif len(box2) == 0:
        box = box1
    else:
        box = np.concatenate([box1, box2], axis=0)
    return image, box


def ricap(
    image1: np.ndarray,
    box1: np.ndarray,
    image2: np.ndarray,
    box2: np.ndarray,
    axis=None,
):
    # 随机选择需要切的轴 x 或 z
    axis = np.random.choice([1, 3], replace=False) if axis is None else axis
    # 相应轴的大小
    axis_size = image1.shape[axis]
    min_offset = 0.4
    # 随机的裁剪位置
    crop_pos = np.random.randint(
        int(axis_size * min_offset), int(axis_size * (1 - min_offset))
    )
    # 拼接图像
    new_image = np.zeros_like(image1)  # shape: [2, D, H, W]
    new_box = []
    if axis == 1:  # z 轴
        new_image[:, :crop_pos, :, :] = image1[:, :crop_pos, :, :]
        new_image[:, crop_pos:, :, :] = image2[:, crop_pos:, :, :]
        # 合并 box
        for b in box1:
            x1, y1, z1, x2, y2, z2, c = b
            if z1 < crop_pos and z2 > crop_pos:
                new_box.append([x1, y1, z1, x2, y2, crop_pos, c])
            elif z2 <= crop_pos:
                new_box.append(b)
        for b in box2:
            x1, y1, z1, x2, y2, z2, c = b
            if z1 < crop_pos and z2 > crop_pos:
                new_box.append([x1, y1, crop_pos, x2, y2, z2, c])
            elif z1 >= crop_pos:
                new_box.append(b)
    elif axis == 3:  # x 轴
        new_image[:, :, :, :crop_pos] = image1[:, :, :, :crop_pos]
        new_image[:, :, :, crop_pos:] = image2[:, :, :, crop_pos:]
        # 合并 box
        for b in box1:
            x1, y1, z1, x2, y2, z2, c = b
            if x1 < crop_pos and x2 > crop_pos:
                new_box.append([x1, y1, z1, crop_pos, y2, z2, c])
            elif x2 <= crop_pos:
                new_box.append(b)
        for b in box2:
            x1, y1, z1, x2, y2, z2, c = b
            if x1 < crop_pos and x2 > crop_pos:
                new_box.append([crop_pos, y1, z1, x2, y2, z2, c])
            elif x1 >= crop_pos:
                new_box.append(b)
    return new_image, np.array(new_box)
