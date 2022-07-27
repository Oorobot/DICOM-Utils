import cv2
from mlxtend.evaluate import mcnemar, mcnemar_table
import pandas as pd
import numpy as np


############################################################
### 独立验证集，提出的最佳模型与医生进行比较，计算显著性水平p ###
############################################################

# knee_ = pd.read_csv("Files/dicom/knee.csv")
# hip_ = pd.read_csv("Files/dicom/hip.csv")
# knee_label = knee_["label"].values
# hip_label = hip_["label"].values
# folds = ["fold 1", "fold 2", "fold 3", "fold 4", "fold 5"]
# doctors = ["D1", "D2", "D3"]
# print("Knee\nFold \tDoctor\tp\tchi2")
# for fold in folds:
#     for doctor in doctors:
#         tb = mcnemar_table(
#             y_target=knee_label,
#             y_model1=knee_[fold].values,
#             y_model2=knee_[doctor].values,
#         )
#         chi2, p = mcnemar(tb)
#         print("{}\t{}\t{:.4f}\t{:.4f}".format(fold, doctor, p, chi2))
# print("Hip\nFold \tDoctor\tp\tchi2")
# for fold in folds:
#     for doctor in doctors:
#         tb = mcnemar_table(
#             y_target=hip_label,
#             y_model1=hip_[fold].values,
#             y_model2=hip_[doctor].values,
#         )
#         chi2, p = mcnemar(tb)
#         print("{}\t{}\t{:.4f}\t{:.4f}".format(fold, doctor, p, chi2))

#############################################
### 独立验证集，制表给医生评估，计算分类指标 ###
#############################################
# from utils.tool import load_json, classification_metrics
# import pandas as pd
# import numpy as np

# knee_ = pd.read_csv("Files/dicom/knee.csv")
# hip_ = pd.read_csv("Files/dicom/hip.csv")
# knee_label = knee_["label"].values
# hip_label = hip_["label"].values

# col_names = ["D1", "D2", "D3", "fold 1", "fold 2", "fold 3", "fold 4", "fold 5"]
# print("Knee")
# print("Doctor\t Acc\t Spec\t Sen\t F1\t PPV\t NPV")
# for col_name in col_names:
#     d = knee_[col_name].values
#     result = classification_metrics(knee_label, d)
#     print(
#         "{}  \t {:.4f}\t {:.4f}\t {:.4f}\t {:.4f}\t {:.4f}\t {:.4f}".format(
#             col_name, *result[1:-1]
#         )
#     )
# print("Hip")
# print("Doctor\t Acc\t Spec\t Sen\t F1\t PPV\t NPV")
# for col_name in col_names:
#     d = hip_[col_name].values
#     result = classification_metrics(hip_label, d)
#     print(
#         "{}  \t {:.4f}\t {:.4f}\t {:.4f}\t {:.4f}\t {:.4f}\t {:.4f}".format(
#             col_name, *result[1:-1]
#         )
#     )


######################################################
### 使用镜像的方式将 SUVmax > 2.5 的数据扩展至原来的四倍
######################################################


z = np.load("Files/PETCT/001_CT131_00.npz")


def normalize(hu: np.ndarray):
    hu = np.clip(hu, -1000, 1000)
    hu = hu / 2000 + 0.5
    return hu


image1 = normalize(z["HU"])
image2 = np.flip(image1, 0)
image3 = np.flip(image1, 1)
image4 = np.flip(image2, 1)
cv2.imshow("img", image1)
cv2.waitKey()
cv2.imshow("img", image2)
cv2.waitKey()
cv2.imshow("img", image3)
cv2.waitKey()
cv2.imshow("img", image4)
cv2.waitKey()
