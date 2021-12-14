import numpy as np
import cv2
import sys
from glob import glob
from utils import *


# 排除异常数据，process/reg\\063_CT286_01.npz', 'process/reg\\063_CT286_02.npz


file = open("data_valid.txt", "w")
sys.stdout = file


# 进行数据校验
npzs = glob("process/reg/*.npz")
suv_max_max = 0
suv_min_min = 1
abnormal_file = []
for npz in npzs:
    print("file name: ", npz)
    data = np.load(npz)
    seg = data["seg"]
    hu = data["hu"]
    suvmax = data["suvmax"]
    suvmin = data["suvmin"]
    suvmean = data["suvmean"]

    contours, h = cv2.findContours(
        seg.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE
    )
    print("the number of segmentation: ", len(contours))
    if len(contours) > 1:
        abnormal_file.append(npz)
    print(
        "the existence of nan and inf (Ture or False) : ",
        np.isnan(hu).any(),
        np.isinf(hu).any(),
        np.isnan(seg).any(),
        np.isinf(seg).any(),
    )
    print("suv max, min, mean: %f, %f, %f" % (suvmax, suvmin, suvmean))
    if suvmax > suv_max_max:
        suv_max_max = suvmax
    if suvmin < suv_min_min:
        suv_min_min = suvmin
print("the max of suv max: ", suv_max_max)
print("the min of suv min: ", suv_min_min)
print("abnormal files: ", abnormal_file)
file.close()

"""以下代码用于填充除中心以外的所有分割标签
"""
# npzs = glob("process/reg/*.npz")
# 076_CT409_00, 076_CT409_02 特殊情况排除掉
# npzs = ["process/reg\\076_CT409_00.npz", "process/reg\\076_CT409_02.npz"]

# for npz in npzs:
#     data = np.load(npz)
#     seg = data["seg"]
#     contours, hierarchy = cv2.findContours(
#         seg.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE
#     )
#     if len(contours) != 1:
#         draw_list = []
#         img = seg.astype(np.uint8)
#         save_image(img, "gray", "process/fill/" + npz.split("\\")[-1][:-4] + ".png")
#         for idx, contour in enumerate(contours):
#             contour = np.squeeze(contour)
#             if len(contour.shape) == 1:
#                 draw_list.append(idx)
#                 continue
#             indices_max = np.max(contour, axis=0)
#             indices_min = np.min(contour, axis=0)
#             if (
#                 indices_min[1] <= 15.5 <= indices_max[1]
#                 and indices_min[0] <= 15.5 <= indices_max[0]
#             ):
#                 pass
#             else:
#                 draw_list.append(idx)
#         for d in draw_list:
#             cv2.drawContours(img, contours, d, (0, 0, 0), cv2.FILLED)
#         save_image(img, "gray", "process/fill/" + npz.split("\\")[-1][:-4] + "__.png")

#         np.savez(
#             "process/fill/" + npz.split("\\")[-1][:-4] + ".npz",
#             hu=data["hu"],
#             seg=img,
#             suvmax=data["suvmax"],
#             suvmean=data["suvmean"],
#             suvmin=data["suvmin"],
#         )
