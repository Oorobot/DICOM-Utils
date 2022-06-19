import os
from glob import glob
from turtle import position

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydicom

from dicom import get_pixel_value
from utils import load_json, mkdir, mkdirs
import cv2

# 读取数据信息
xlsx = pd.read_excel("ThreePhaseBone/ThreePhaseBone.xlsx")
infos = xlsx[["编号", "最终结果", "部位", "type"]].values

knee = "_ProcessedData_/knee/001_DCM.npz"
hip = "_ProcessedData_/hip/003_DCM.npz"
hip_ = "_ProcessedData_/hip_/003_DCM.npz"
hip_focus_mask = "ThreePhaseBone/hip_focus/003/Untitled.nii.gz"
hip_focus_left = "_ProcessedData_/hip_focus/236_r_1.npz"
hip_focus_right = "_ProcessedData_/hip_focus/236_r_1.npz"
normal_hip = np.load("normal_hip.npz")

save_dir = "_ProcessedData_/pic/"


def normalized(data: np.ndarray, ratio: float = 0.5, mask: np.ndarray = None):
    flow_max = np.max(data[00:20]) * ratio
    pool_max = np.max(data[20:25]) * ratio
    if mask is not None:
        data = data * (1 - mask) + data * (mask / 6.0)
    normalized_flow = data[00:20] / flow_max
    normalized_pool = data[20:25] / pool_max
    return np.concatenate((normalized_flow, normalized_pool), axis=0)


# knee
# data = np.load(knee)["data"]
# data = normalized(data, 0.5)
# # 保存图片
mask = np.load("0_0.npy")
heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
# heatmap = np.float32(heatmap) / 255
# for i in range(25):

#     img = 1 - data[i]
#     img = img[:, :, np.newaxis]

#     new_img = np.repeat(img, repeats=3, axis=2)

#     cv2.namedWindow("new_image")
#     cv2.imshow("new_image", new_img)

#     cam = heatmap + img
#     cam = cam / np.max(cam)
#     cv2.namedWindow("image")
#     cv2.imshow("image", cam)
#     cv2.waitKey()
#     cv2.destroyAllWindows()
#     cv2.imwrite(f"{i}.jpg", cam * 255)
# plt.imshow(data[i], plt.cm.binary)
# plt.axis("off")
# plt.savefig(
#     save_dir + f"knee{str(i+1).zfill(2)}.png", bbox_inches="tight", pad_inches=-0.1
# )
# plt.close()


# hip
data = np.load(hip_focus_left)["data"]
data = normalized(data, 0.5)
boundary = np.load(hip_focus_left)["boundary"]
mask = np.load("1_1.npy")
heatmap = cv2.applyColorMap(np.uint8(255 * np.squeeze(mask)), cv2.COLORMAP_JET)
heatmap = np.float32(heatmap) / 255
# 保存图片
for i in range(25):
    img = 1 - data[i]
    img = img[:, :, np.newaxis]

    new_img = np.repeat(img, repeats=3, axis=2)

    cam = (
        heatmap
        + new_img[boundary[0] : boundary[1] + 1, boundary[2] : boundary[3] + 1, :]
    )
    cam = cam / np.max(cam)
    new_img[boundary[0] : boundary[1] + 1, boundary[2] : boundary[3] + 1, :] = cam
    cv2.namedWindow("image")
    cv2.imshow("image", new_img)
    cv2.waitKey()
    cv2.destroyAllWindows()
    cv2.imwrite(f"{i}.jpg", new_img * 255)

# hip_
# data = np.load(hip_)
# mask = data["mask"]
# data = data["data"]
# data = normalized(data, 0.5)
# plt.imshow(data[-1], plt.cm.binary)
# plt.imshow(mask, plt.cm.Greens, alpha=0.5)
# plt.axis("off")
# plt.savefig(save_dir + f"hip_mask.png", bbox_inches="tight", pad_inches=-0.1)
# plt.close()


# # 保存图片
# for i in range(25):
#     plt.imshow(data[i], plt.cm.binary)
#     plt.axis("off")
#     plt.savefig(
#         save_dir + f"hip_{str(i+1).zfill(2)}.png", bbox_inches="tight", pad_inches=-0.1
#     )
#     plt.close()

# # hip_focus
# left_normal_hip = normal_hip["left"]
# right_normal_hip = normal_hip["right"]
# left_normal_hip = normalized(left_normal_hip, 1.0)
# right_normal_hip = normalized(right_normal_hip, 1.0)
# for i in range(25):
#     plt.imshow(left_normal_hip[i], plt.cm.binary)
#     plt.axis("off")
#     plt.savefig(
#         save_dir + f"normal_left_hip_{str(i+1).zfill(2)}.png",
#         bbox_inches="tight",
#         pad_inches=-0.1,
#     )
#     plt.imshow(right_normal_hip[i], plt.cm.binary)
#     plt.axis("off")
#     plt.savefig(
#         save_dir + f"normal_right_hip_{str(i+1).zfill(2)}.png",
#         bbox_inches="tight",
#         pad_inches=-0.1,
#     )
#     plt.close()

# focus_mask = get_pixel_value(hip_focus_mask)
# plt.imshow(focus_mask[24], plt.cm.gray)
# plt.axis("off")
# plt.savefig(save_dir + f"hip_focus_mask.png", bbox_inches="tight", pad_inches=-0.1)
# plt.close()

# left = np.load(hip_focus_left)["data"]
# boundary_left = np.load(hip_focus_left)["boundary"]
# processed_left = (
#     left[
#         :,
#         boundary_left[0] : boundary_left[1] + 1,
#         boundary_left[2] : boundary_left[3] + 1,
#     ]
#     - normal_hip["left"]
# )
# processed_left = processed_left - np.min(processed_left) / (
#     np.max(processed_left) - np.min(processed_left)
# )
# right = np.load(hip_focus_right)["data"]
# boundary_right = np.load(hip_focus_right)["boundary"]
# processed_right = (
#     right[
#         :,
#         boundary_right[0] : boundary_right[1] + 1,
#         boundary_right[2] : boundary_right[3] + 1,
#     ]
#     - normal_hip["right"]
# )
# processed_right = processed_right - np.min(processed_right) / (
#     np.max(processed_right) - np.min(processed_right)
# )
# for i in range(25):
#     plt.imshow(processed_left[i], plt.cm.binary)
#     plt.axis("off")
#     plt.savefig(
#         save_dir + f"left_processed_hip_{str(i+1).zfill(2)}.png",
#         bbox_inches="tight",
#         pad_inches=-0.1,
#     )
#     plt.imshow(
#         left[
#             i,
#             boundary_left[0] : boundary_left[1] + 1,
#             boundary_left[2] : boundary_left[3] + 1,
#         ],
#         plt.cm.binary,
#     )
#     plt.axis("off")
#     plt.savefig(
#         save_dir + f"left_hip_{str(i+1).zfill(2)}.png",
#         bbox_inches="tight",
#         pad_inches=-0.1,
#     )
#     plt.imshow(processed_right[i], plt.cm.binary)
#     plt.axis("off")
#     plt.savefig(
#         save_dir + f"right_processed_hip_{str(i+1).zfill(2)}.png",
#         bbox_inches="tight",
#         pad_inches=-0.1,
#     )
#     plt.imshow(
#         right[
#             i,
#             boundary_right[0] : boundary_right[1] + 1,
#             boundary_right[2] : boundary_right[3] + 1,
#         ],
#         plt.cm.binary,
#     )
#     plt.axis("off")
#     plt.savefig(
#         save_dir + f"right_hip_{str(i+1).zfill(2)}.png",
#         bbox_inches="tight",
#         pad_inches=-0.1,
#     )
#     plt.close()

# hip with boundary
data = np.load(hip)["data"]
data = normalized(data, 0.5)
boundary_left = np.load(hip_focus_left)["boundary"]
boundary_right = np.load(hip_focus_right)["boundary"]
# data[
#     -1,
#     boundary_right[0] : boundary_right[1] + 1,
#     boundary_right[2] : boundary_right[3] + 1,
# ] = 0


# # 保存图片
# for i in range(25):
#     plt.imshow(data[i], plt.cm.binary)
#     plt.gca().add_patch(
#         plt.Rectangle(
#             (boundary_right[2], boundary_right[0]),
#             boundary_right[3] + 1 - boundary_right[2],
#             boundary_right[1] + 1 - boundary_right[0],
#             linewidth=1,
#             edgecolor="r",
#             facecolor="none",
#         )
#     )
#     plt.gca().add_patch(
#         plt.Rectangle(
#             (boundary_left[2], boundary_left[0]),
#             boundary_left[3] + 1 - boundary_left[2],
#             boundary_left[1] + 1 - boundary_left[0],
#             linewidth=1,
#             edgecolor="r",
#             facecolor="none",
#         )
#     )
#     plt.axis("off")
#     plt.savefig(
#         save_dir + f"hip{str(i+1).zfill(2)}_with_boundary.png",
#         bbox_inches="tight",
#         pad_inches=-0.1,
#     )
#     plt.close()


################################################
### 独立验证集，制表给医生评估，计算分类指标
################################################

# knee_validate = load_json("_ProcessedData_/dicom_knee_2.json")["fold_1"]["validate"]
# hip_validate = load_json("_ProcessedData_/dicom_hip_focus12_5.json")["fold_1"][
#     "validate"
# ]


# mkdirs(
#     ["_ProcessedData_/dicom", "_ProcessedData_/dicom/hip", "_ProcessedData_/dicom/knee"]
# )

# import pandas as pd

# origin_k = []
# changed_k = []

# knee_validate = np.sort(knee_validate)
# knee_label = [
#     np.load(k.replace("data", "_ProcessedData_"))["label"] for k in knee_validate
# ]
# for i, k in enumerate(knee_validate):
#     num = os.path.basename(k)[0:3]
#     dir = "ThreePhaseBone/knee"

#     # 清除隐私信息
#     filename = os.path.join(dir, num, f"{num}_FLOW.dcm")
#     file = pydicom.dcmread(filename)
#     file.InstitutionName = ""
#     file.InstitutionAddress = ""
#     file.OperatorName = ""
#     file.ReferrringPhysicianName = ""
#     file.PatientID = ""
#     file.PatientName = ""

#     # 对文件名加密

#     save_file_name = str(i).zfill(2)

#     origin_k.append(num)
#     changed_k.append(save_file_name)
#     print(num + "-" + save_file_name)
#     # 保存文件
#     mkdir(f"_ProcessedData_/dicom/knee/{save_file_name}")
#     file.save_as(f"_ProcessedData_/dicom/knee/{save_file_name}/{save_file_name}.dcm")
# dataframe = pd.DataFrame(
#     {"origin_file": origin_k, "file": changed_k, "label": knee_label}
# )
# dataframe.to_csv(f"_ProcessedData_/dicom/knee/knee_with_label.csv")

# origin_h = []
# changed_h = []
# pos = []

# hip_validate = np.sort(hip_validate)
# hip_label = [
#     np.load(h.replace("data", "_ProcessedData_"))["label"] - 1 for h in hip_validate
# ]
# for i, h in enumerate(hip_validate):
#     filename = os.path.basename(h)
#     num = filename[0:3]
#     position = filename[4]

#     dir = "ThreePhaseBone/hip"

#     # 清除隐私信息
#     filename = os.path.join(dir, num, f"{num}_FLOW.dcm")
#     file = pydicom.dcmread(filename)
#     file.InstitutionName = ""
#     file.InstitutionAddress = ""
#     file.OperatorName = ""
#     file.ReferrringPhysicianName = ""
#     file.PatientID = ""
#     file.PatientName = ""

#     # 对文件名加密
#     save_file_name = str(i).zfill(2)
#     origin_h.append(num)
#     changed_h.append(save_file_name)
#     pos.append(position)
#     print("-".join([num, save_file_name, position]))
#     mkdir(f"_ProcessedData_/dicom/hip/{save_file_name}")
#     file.save_as(f"_ProcessedData_/dicom/hip/{save_file_name}/{save_file_name}.dcm")
# dataframe = pd.DataFrame(
#     {"origin_file": origin_h, "file": changed_h, "position": pos, "label": hip_label}
# )
# dataframe.to_csv(f"_ProcessedData_/dicom/hip/hip_with_label.csv")


# import pandas as pd
# from sklearn.metrics import confusion_matrix


# knee_true = pd.read_csv("_ProcessedData_/dicom/knee/knee_with_label.csv")[
#     "label"
# ].values
# hip_true = pd.read_csv("_ProcessedData_/dicom/hip/hip_with_label.csv")["label"].values


# def metric(file_name: str, col_name: str, log):
#     pred = pd.read_excel(file_name)[col_name].values
#     file_name = os.path.basename(file_name)
#     if file_name.find("knee") != -1:
#         tn, fp, fn, tp = confusion_matrix(knee_true, pred).ravel()
#     else:
#         tn, fp, fn, tp = confusion_matrix(hip_true, pred).ravel()
#     accuracy = (tn + tp) / (tn + fp + fn + tp)
#     precision = tp / (tp + fp)
#     sensitivity = tp / (tp + fn)
#     specificity = tn / (tn + fp)
#     f1 = 2 * (precision * sensitivity) / (precision + sensitivity)

#     log.write(file_name + "\n")
#     log.write("{} {}\n{} {}\n".format(tn, fp, fn, tp))
#     log.write(
#         "[Accuracy] {0:5.2%}({1:2d}/{2:2d}) [Precision] {3:5.2%}({4:2d}/{5:2d}) [sensitivity] {6:5.2%}({7:2d}/{8:2d}) [specificity] {9:5.2%}({10:2d}/{11:2d}) [F1 Score] {12:.4f}\n".format(
#             accuracy,
#             tn + tp,
#             tn + fp + fn + tp,
#             precision,
#             tp,
#             tp + fp,
#             sensitivity,
#             tp,
#             tp + fn,
#             specificity,
#             tn,
#             tn + tp,
#             f1,
#         )
#     )
#     log.flush()


# log_metric = open("_ProcessedData_/dicom/log_metric.log", "w+")
# metric("_ProcessedData_/dicom/knee/D-knee.xlsx", "最终结果（未感染-0，感染-1）", log_metric)
# metric("_ProcessedData_/dicom/knee/S-knee.xlsx", "最终结果（未感染-0，感染-1）", log_metric)
# metric("_ProcessedData_/dicom/knee/W-knee.xlsx", "最终结果（未感染-0，感染-1）", log_metric)
# metric("_ProcessedData_/dicom/hip/D-hip.xlsx", "判断结果（未感染-0，感染-1）", log_metric)
# metric("_ProcessedData_/dicom/hip/S-hip.xlsx", "判断结果（未感染-0，感染-1）", log_metric)
# metric("_ProcessedData_/dicom/hip/W-hip.xlsx", "判断结果（未感染-0，感染-1）", log_metric)
# log_metric.close()

