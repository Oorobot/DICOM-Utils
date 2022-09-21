import json
import os
import stat

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
