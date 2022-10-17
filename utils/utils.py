import json
import os
import stat
from glob import glob
from turtle import width
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from xpinyin import Pinyin

# 常量
OUTPUT_FOLDER = "./Files"


# 工具
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


def is_empty(dir: str):
    return not os.listdir(str)


# json
def save_json(save_path: str, data: dict):
    assert save_path.split(".")[-1] == "json"
    with open(save_path, "w") as file:
        json.dump(data, file)


def load_json(file_path: str):
    assert file_path.split(".")[-1] == "json"
    with open(file_path, "r") as file:
        data = json.load(file)
    return data


# 字符串去除前后空格 str.strip()

# 汉字转拼音
P = Pinyin()


def to_pinyin(chinese_characters: str):
    return P.get_pinyin(chinese_characters.strip(), " ", convert="upper")


# 画直方图
COLORS = ["#63b2ee", "#76da91", "#f8cb7f"]


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
