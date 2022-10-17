import shutil
from tkinter import font
from utils.utils import load_json, save_json
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import zipfile

# data_files = load_json("2022-10-01.json")
# five_fold = load_json("five_fold.json")
# independent = load_json("independent.json")
# pulmonary_nodules = load_json("pulmonary_nodules.json")

# model_output = {"test": five_fold, "validate": independent}
# res = {"test": {}, "validate": {}}
# for fold_name, each_fold in data_files.items():
#     for t in ["test", "validate"]:
#         files = each_fold[t]
#         for i, file in enumerate(files):
#             filename = os.path.basename(file)
#             names = os.path.splitext(filename)[0].split("_")
#             preprocessed_data_suv = pulmonary_nodules[names[0]][
#                 names[1] + "_" + names[2]
#             ]["suv"]
#             distance = pulmonary_nodules[names[0]][names[1] + "_" + names[2]][
#                 "distance"
#             ]
#             ground_truth_suv = model_output[t][fold_name]["ground_truth"][i]
#             equal = np.equal(ground_truth_suv[0], preprocessed_data_suv[0])
#             if not equal.all():
#                 print(fold_name, t, filename)
#             predicted_suv = model_output[t][fold_name]["prediction"][i]

#             if fold_name not in res[t]:
#                 res[t][fold_name] = {}
#             res[t][fold_name]["_".join(names)] = {
#                 "distance": distance,
#                 "ground_truth": ground_truth_suv,
#                 "prediction": predicted_suv,
#                 "absolute_error": np.abs(
#                     np.array(ground_truth_suv) - np.array(predicted_suv)
#                 ).tolist(),
#             }


# res = load_json("res.json")
# five_fold, ind = res["test"], res["validate"]


# titles = ["Five-Fold Cross-Validation", "Independet Test"]
# for j, data in enumerate([five_fold, ind]):
#     sorted_by_distance = []
#     sorted_by_suvmax = []
#     for i, (fold_name, each_fold) in enumerate(data.items()):
#         suvmax, distance, error = [], [], []
#         for filename, attr in each_fold.items():
#             error.append(attr["absolute_error"][0])
#             suvmax.append(attr["ground_truth"][0])
#             distance.append(attr["distance"])

#         indices = np.argsort(suvmax)
#         sorted_error = np.array(error)[indices]
#         sorted_suvmax = np.array(suvmax)[indices]
#         sorted_by_suvmax.append([sorted_suvmax, sorted_error])

#         indices = np.argsort(distance)
#         sorted_error = np.array(error)[indices]
#         sorted_distance = np.array(distance)[indices]
#         sorted_by_distance.append([sorted_distance, sorted_error])

#     plt.figure(figsize=(12.8, 7.2), dpi=100)
#     colors = ["#008000", "#ff4d00", "#470024", "#a16b47", "#00008b"]
#     for i, sorted in enumerate(sorted_by_suvmax):
#         plt.scatter(
#             sorted[0], sorted[1], label=f"Fold {i+1}", color=colors[i], s=1,
#         )
#     plt.legend(fontsize=8, ncol=5)
#     plt.title(
#         titles[j], fontdict={"fontsize": 18, "family": "Times New Roman"},
#     )
#     plt.xlabel(
#         "Groud Truth: SUVmax", fontdict={"fontsize": 14, "family": "Times New Roman"}
#     )
#     plt.ylabel("Absolute Error", fontdict={"fontsize": 14, "family": "Times New Roman"})
#     plt.xticks(fontsize=8)
#     plt.yticks(fontsize=8)
#     plt.xlim(0, 60)
#     plt.savefig(f"Figure {2*j+1}.svg")

#     plt.figure(figsize=(12.8, 7.2), dpi=100)
#     colors = ["#008000", "#ff4d00", "#470024", "#a16b47", "#00008b"]
#     for i, sorted in enumerate(sorted_by_distance):
#         plt.scatter(
#             sorted[0], sorted[1], label=f"Fold {i+1}", color=colors[i], s=1,
#         )
#     plt.legend(fontsize=8, ncol=5)
#     plt.title(
#         titles[j], fontdict={"fontsize": 18, "family": "Times New Roman"},
#     )
#     plt.xlabel(
#         "Maximum Diameter", fontdict={"fontsize": 14, "family": "Times New Roman"}
#     )
#     plt.ylabel("Absolute Error", fontdict={"fontsize": 14, "family": "Times New Roman"})
#     plt.xticks(fontsize=8)
#     plt.yticks(fontsize=8)
#     plt.xlim(0, 60)
#     plt.savefig(f"Figure {2*j+2}.svg")

# data_file = load_json("Files/2022-10-01.json")
# diff = np.setdiff1d(data_file["fold_1"]["validate"], data_file["fold_2"]["validate"])
# same_files = np.setdiff1d(data_file["fold_1"]["validate"], diff).tolist()
# summary = load_json("Files/res.json")
# five_fold = summary["test"]
# independent = summary["validate"]
# col_names = [
#     "Maximum Diameter",
#     "SUVmax",
#     "SUVmean",
#     "SUVmin",
#     "predicted SUVmax",
#     "predicted SUVmean",
#     "predicted SUVmin",
#     "绝对误差 SUVmax",
#     "绝对误差 SUVmean",
#     "绝对误差 SUVmin",
#     "平均绝对误差",
#     "相对误差 SUVmax",
#     "相对误差 SUVmean",
#     "相对误差 SUVmin",
#     "平均相对误差",
#     "平方差 SUVmax",
#     "平方差 SUVmean",
#     "平方差 SUVmin",
#     "平均平方差",
# ]

# # excel_writer = pd.ExcelWriter("five_fold.xlsx")
# with pd.ExcelWriter("independent.xlsx") as writer:
#     r = pd.DataFrame(columns=col_names)
#     index_name = []
#     for file in same_files:
#         key = os.path.basename(file)[:-4]
#         index_name.append(key)
#         diameter = []
#         s_g = []
#         s_p = []
#         info = []
#         for fold_name, value in independent.items():
#             diameter.append(value[key]["distance"])
#             s_g.append(value[key]["ground_truth"])
#             s_p.append(value[key]["prediction"])
#         diameter = np.mean(diameter)
#         s_g = np.mean(s_g, axis=0)
#         s_p = np.mean(s_p, axis=0)
#         info = info + [diameter] + s_g.tolist() + s_p.tolist()
#         ae = np.abs(s_g - s_p)
#         re = ae / s_g
#         se = ae ** 2

#         def add(i, e):
#             return i + e.tolist() + [np.mean(e)]

#         info = add(info, ae)
#         info = add(info, re)
#         info = add(info, se)
#         r.loc[len(r)] = info

#     r.index = index_name
#     r.to_excel(writer, sheet_name="average")

#     for fold_name, value in independent.items():
#         r = pd.DataFrame(columns=col_names)
#         index_name = []

#         for k, v in value.items():
#             index_name.append(k)
#             info = []
#             d = v["distance"]
#             suv = v["ground_truth"]
#             pred_suv = v["prediction"]
#             info = info + [d] + suv + pred_suv
#             suv, pred_suv = np.array(suv), np.array(pred_suv)
#             ae = np.abs(suv - pred_suv)
#             re = ae / suv
#             se = ae ** 2

#             def add(i, e):
#                 return i + e.tolist() + [np.mean(e)]

#             info = add(info, ae)
#             info = add(info, re)
#             info = add(info, se)
#             r.loc[len(r)] = info

#         r.index = index_name
#         r.to_excel(writer, sheet_name=fold_name)
