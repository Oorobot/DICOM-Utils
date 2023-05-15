import copy
import datetime
import math
import os
import shutil
import zipfile
from glob import glob

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydicom
import SimpleITK as sitk
from mlxtend.evaluate import mcnemar, mcnemar_table  # 用于计算显著性水平 p
from skimage import measure

from utils.dicom import (
    normalize_ct_hu,
    normalize_pet_suv,
    read_serises_image,
    resameple_based_size,
    resample_based_spacing,
    resample_based_target_image,
)
from utils.html import HTML
from utils.utils import delete, load_json, mkdir, rename, save_json, to_pinyin


"""
    可视化探测结果
"""
dt_dir = "Files/FRI/01-b6-n1-5-mixup0.5-rol"
annotations = load_json("Files/FRI/image_2mm.json")
str2int = {
    "infected": 1,
    "uninfected": 2,
    "bladder": 3,
    "other": 4,
    "lesion": 5,
}
int2str = {
    1: "GT infected",
    2: "GT uninfected",
    3: "GT bladder",
    5: "GT lesion",
    6: "DT infected",
    7: "DT uninfected",
    8: "DT bladder",
    10: "DT lesion",
}
int2color = {
    # BGR
    1: (0, 0, 1),  # 红色
    2: (0, 1, 0),  # 绿色
    3: (1, 0, 0),  # 蓝色
    5: (0, 0, 1),  # 红色
    6: (0.5451, 0, 1),
    7: (0, 0.8431, 1),
    8: (1, 0.4510, 0),
    10: (63 / 255, 133 / 255, 205 / 255),  # 蓝品红
}


def draw_label(img, b):
    label_text = int2str[b[0]]
    label_color = int2color[b[0]]
    cv2.rectangle(img, b[1:3], b[3:5], color=label_color, thickness=1)
    labelSize = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.3, 1)[0]
    if b[0] > 5:  # 文字放框上
        cv2.putText(
            img,
            label_text,
            (b[1], b[2] - 1),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.3,
            color=label_color,
            thickness=1,
        )
    else:  # 文字放框下
        cv2.putText(
            img,
            label_text,
            (b[1], b[4] + labelSize[1] + 1),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.3,
            color=label_color,
            thickness=1,
        )


def save_3d_label(no, gts, dts):
    object_image = sitk.ReadImage(
        os.path.join("Files/FRI/image_2mm", f"{no}_Label_Object.nii.gz")
    )
    object_array = sitk.GetArrayFromImage(object_image)
    dt_array = np.zeros_like(object_array)
    image_shape = dt_array.shape[::-1]
    for gt in gts:
        for i in range(3):
            gt[i + 4] -= 1
        c, x1, y1, z1, x2, y2, z2 = gt
        if c == 4:
            continue
        xy = np.full((y2 - y1, x2 - x1), c)
        dt_array[z1, y1:y2, x1:x2] = xy
        dt_array[z2, y1:y2, x1:x2] = xy
        xz = np.full((z2 - z1, x2 - x1), c)
        dt_array[z1:z2, y1, x1:x2] = xz
        dt_array[z1:z2, y2, x1:x2] = xz
        yz = np.full((z2 - z1, y2 - y1), c)
        dt_array[z1:z2, y1:y2, x1] = yz
        dt_array[z1:z2, y1:y2, x2] = yz
    for dt in dts:
        for i in range(3):
            dt[i + 1] = max(dt[i + 1], 0)
            dt[i + 4] = min(dt[i + 4], image_shape[i]) - 1
        c, x1, y1, z1, x2, y2, z2 = dt
        xy = np.full((y2 - y1, x2 - x1), c + 5)
        dt_array[z1, y1:y2, x1:x2] = xy
        dt_array[z2, y1:y2, x1:x2] = xy
        xz = np.full((z2 - z1, x2 - x1), c + 5)
        dt_array[z1:z2, y1, x1:x2] = xz
        dt_array[z1:z2, y2, x1:x2] = xz
        yz = np.full((z2 - z1, y2 - y1), c + 5)
        dt_array[z1:z2, y1:y2, x1] = yz
        dt_array[z1:z2, y1:y2, x2] = yz
    # 保存.nii.gz图像
    dt_image = sitk.GetImageFromArray(dt_array)
    sitk.WriteImage(dt_image, f"Files/FRI/dts/{no}.nii.gz")


if __name__ == '__main__':
    gray_cmap = matplotlib.colormaps["gray"]
    gray_colors = gray_cmap(np.linspace(0, 1, gray_cmap.N))[:, :3]
    hot_cmap = matplotlib.colormaps["hot"]
    hot_colors = hot_cmap(np.linspace(0, 1, hot_cmap.N))[:, :3]

    for file in os.listdir(dt_dir):
        no = file.split(".")[0]
        # ground truth
        labels = annotations[no]["labels"]
        gts = [[str2int[label["category"]]] + label["position"] for label in labels]
        with open(os.path.join(dt_dir, file)) as f:
            lines = f.readlines()
        # detection result
        dts = []
        for line in lines:
            c, _, *bb = line.strip().split()
            dts.append([str2int[c]] + [int(v) for v in bb])
        # 保存三维检测结果
        if False:
            save_3d_label(no, gts, dts)
        # 保存 xz 平面图像, 求病灶间 y 区间的最小交集
        for dt in dts:
            dt[0] += 5
        ts = gts + dts
        ts.sort(key=lambda x: x[2])
        union, tmp = [], None
        for t in ts:
            c, x1, y1, z1, x2, y2, z2 = t
            if tmp is None:
                tmp = [y1, y2, [[c, x1, z1, x2, z2]]]
            else:
                ma = max(tmp[0], y1)
                mi = min(tmp[1], y2)
                if ma < mi:
                    tmp[0] = ma
                    tmp[1] = mi
                    tmp[2].append([c, x1, z1, x2, z2])
                else:
                    union.append(tmp)
                    tmp = [y1, y2, [[c, x1, z1, x2, z2]]]
        if tmp is not None:
            union.append(tmp)
        # 读取 CT 和 SUV
        ct_image = sitk.ReadImage(
            os.path.join("Files/FRI/image_2mm", f"{no}_CT.nii.gz")
        )
        suv_image = sitk.ReadImage(
            os.path.join("Files/FRI/image_2mm", f"{no}_SUVbw.nii.gz")
        )
        ct_array = sitk.GetArrayFromImage(ct_image)
        suv_array = sitk.GetArrayFromImage(suv_image)
        hu = normalize_ct_hu(ct_array, 300, 1500, True)
        suvbw = normalize_pet_suv(suv_array, 2.5, True)

        for i, u in enumerate(union):
            y = (int)((u[0] + u[1]) / 2)
            hu_slice = hu[:, y, :]
            suvbw_slice = suvbw[:, y, :]

            hu_image = gray_colors[hu_slice]
            suv_image = gray_colors[255 - suvbw_slice]
            hot_suv_image = hot_colors[suvbw_slice]
            # BGR -> RGB
            suv_image = suv_image[:, :, ::-1]
            hot_suv_image = hot_suv_image[:, :, ::-1]
            petct = cv2.addWeighted(hu_image, 0.7, hot_suv_image, 0.3, 0)
            hu_image = np.ascontiguousarray(np.flip(hu_image, 0))
            suv_image = np.ascontiguousarray(np.flip(suv_image, 0))
            petct = np.ascontiguousarray(np.flip(petct, 0))

            max_z = hu_image.shape[0] - 1
            for b in u[2]:
                if b[0] == 4:
                    continue
                tmp = max_z - b[4]
                b[4] = max_z - b[2]
                b[2] = tmp
                draw_label(hu_image, b)
                draw_label(suv_image, b)
                draw_label(petct, b)
            img = np.hstack([hu_image, suv_image, petct])
            # cv2.imshow("img", img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            cv2.imwrite(
                f"Files/FRI/dts/{os.path.basename(dt_dir)}-{no}-{i}.png",
                img * 255,
            )

    html = HTML("骨折相关感染", "Files/FRI/dts")
    vals = open("Files/FRI/val.txt", 'r').readlines()
    nos = [val.strip() for val in vals]
    nos.sort()
    for no in nos:
        html.add_header(no)
        imgs = glob(f"Files/FRI/dts/01-b6-n1-5-mixup0.5-rol-{no}*")
        imgs.sort()
        titles = []
        ims = []
        for img in imgs:
            img_name = os.path.basename(img)
            ims.append(img_name)
            titles.append(img_name.split(".")[0])
        html.add_images(ims, titles)
    html.save()
