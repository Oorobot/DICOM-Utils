import copy
import datetime
import math
import os
import shutil
import zipfile
from glob import glob

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydicom
import SimpleITK as sitk
from matplotlib.cm import get_cmap
from mlxtend.evaluate import mcnemar, mcnemar_table  # 用于计算显著性水平 p
from skimage import measure
from utils.html import HTML

from utils.dicom import (
    HU2image,
    SUVbw2image,
    read_serises_image,
    resameple_based_size,
    resample_based_spacing,
    resample_based_target_image,
)
from utils.utils import delete, load_json, mkdir, rename, save_json, to_pinyin


def draw_label_type(draw_img, bbox, label, label_color, up: bool):
    labelSize = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.3, 1)[0]
    if up:
        cv2.putText(
            draw_img,
            label,
            (bbox[0], bbox[1] - 1),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.3,
            color=label_color,
            thickness=1,
        )
    else:
        cv2.putText(
            draw_img,
            label,
            (bbox[0], bbox[3] + labelSize[1] + 1),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.3,
            color=label_color,
            thickness=1,
        )


"""
    可视化探测结果
"""
dt_dir = "Files/01b1_ep780"
annotations = load_json("Files/image_2mm.json")
str2int = {"infected_lesion": 1, "uninfected_lesion": 2, "bladder": 3, "other": 4}
int2str = {
    1: "GT infected",
    2: "GT uninfected",
    3: "GT bladder",
    5: "DT infected",
    6: "DT uninfected",
    7: "DT bladder",
}
int2color = {
    3: (255, 0, 0),  # 红色
    2: (0, 255, 0),  # 绿色
    1: (0, 0, 255),  # 蓝色
    7: (255, 115, 0),
    6: (0, 215, 255),
    5: (139, 0, 255),
}
gray_cmap = get_cmap("gray")
gray_colors = gray_cmap(np.linspace(0, 1, gray_cmap.N))[:, :3]
hot_cmap = get_cmap("hot")
hot_colors = hot_cmap(np.linspace(0, 1, hot_cmap.N))[:, :3]


for file in os.listdir(dt_dir):
    no = file.split(".")[0]
    with open(os.path.join(dt_dir, file)) as f:
        lines = f.readlines()
    dts = []
    for line in lines:
        c, _, *bb = line.strip().split()
        dts.append([str2int[c]] + [int(v) for v in bb])
    labels = annotations[no]["labels"]
    gts = [[str2int[label["category"]]] + label["position"] for label in labels]
    object_image = sitk.ReadImage(
        os.path.join("Files/val_2mm", f"{no}_Label_Object.nii.gz")
    )
    object_array = sitk.GetArrayFromImage(object_image)
    dt_array = np.zeros_like(object_array)
    upper = dt_array.shape[::-1]
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
            dt[i + 4] = min(dt[i + 4], upper[i]) - 1
        c, x1, y1, z1, x2, y2, z2 = dt
        xy = np.full((y2 - y1, x2 - x1), c + 4)
        dt_array[z1, y1:y2, x1:x2] = xy
        dt_array[z2, y1:y2, x1:x2] = xy
        xz = np.full((z2 - z1, x2 - x1), c + 4)
        dt_array[z1:z2, y1, x1:x2] = xz
        dt_array[z1:z2, y2, x1:x2] = xz
        yz = np.full((z2 - z1, y2 - y1), c + 4)
        dt_array[z1:z2, y1:y2, x1] = yz
        dt_array[z1:z2, y1:y2, x2] = yz
    # 保存.nii.gz图像
    # dt_image = sitk.GetImageFromArray(dt_array)
    # sitk.WriteImage(dt_image, f"Files/val_2mm/{no}.nii.gz")
    # 保存 xz 平面图像, 求病灶间 y 区间的最小交集
    for dt in dts:
        dt[0] += 4
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
    ct_image = sitk.ReadImage(os.path.join("Files/val_2mm", f"{no}_CT.nii.gz"))
    suv_image = sitk.ReadImage(os.path.join("Files/val_2mm", f"{no}_SUVbw.nii.gz"))
    ct_array = sitk.GetArrayFromImage(ct_image)
    suv_array = sitk.GetArrayFromImage(suv_image)
    hu = HU2image(ct_array, 300, 1500, True)
    print(np.max(suv_array), np.min(suv_array))
    suvbw = SUVbw2image(suv_array, 2.5, True)

    # suv_array = suv_array[suv_array > 0]
    ex_suvbw = (suv_array - np.mean(suv_array)) / (np.var(suv_array) + 1e-5)
    print(np.max(ex_suvbw), np.min(ex_suvbw))

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
            cv2.rectangle(hu_image, b[1:3], b[3:5], color=int2color[b[0]], thickness=1)
            draw_label_type(hu_image, b[1:5], int2str[b[0]], int2color[b[0]], b[0] <= 4)
            cv2.rectangle(suv_image, b[1:3], b[3:5], color=int2color[b[0]], thickness=1)
            draw_label_type(
                suv_image, b[1:5], int2str[b[0]], int2color[b[0]], b[0] <= 4
            )
            cv2.rectangle(petct, b[1:3], b[3:5], color=int2color[b[0]], thickness=1)
            draw_label_type(petct, b[1:5], int2str[b[0]], int2color[b[0]], b[0] <= 4)
        img = np.hstack([hu_image, suv_image, petct])
        # cv2.imshow("img", img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        cv2.imwrite(
            f"Files/val_2mm_jpg/{os.path.basename(dt_dir)}.{no}.{i}.png", img * 255
        )


"""
    网页生成，查看图片
"""

html = HTML("骨折相关感染", "Files", "val_2mm_jpg")
vals = open("Files/val.txt", 'r').readlines()
nos = [val.strip() for val in vals]
for no in nos:
    html.add_header(no)
    imgs = glob(f"Files/val_2mm_jpg/*.{no}.*")
    imgs.sort()
    titles = []
    ims = []
    for img in imgs:
        img_name = os.path.basename(img)
        ims.append(img_name)
        titles.append(img_name.split(".")[0])
    html.add_images(ims, titles)
html.save()
