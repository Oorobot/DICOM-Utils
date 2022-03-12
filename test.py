# from glob import glob

# import cv2
# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd

# file = "ThreePhaseBone/055/055_FLOW.dcm"

# dicom = dcmread(file)
# window_center, window_width = float(dicom.WindowCenter), float(dicom.WindowWidth)
# pixel_value = get_pixel_value(file)
# gray_image = convert_dicom_to_gary_image_by_window_center_and_window_width(
#     pixel_value, window_center, window_width, to_uint8=True, ratio=0.5
# )
# other_method = np.clip(pixel_value, 0, np.max(pixel_value[0:20]) * 0.5) / (
#     np.max(pixel_value[0:20]) * 0.5
# )
# other_method = (other_method * 255).astype(np.uint8)
# # flip_other_method = np.flip(other_method, -1)

# for i in range(25):
#     cv2.imwrite(f"ProcessedData/flow_gray_{i}.png", other_method[i])


# 将PETCT回归预测数据，读取进行预处理后写成csv文件
import enum
from glob import glob
import numpy as np
import pandas as pd

from utils import load_json

# all_file = glob("ProcessedData/regression_new/*.npz")
# for i in range(len(all_file)):
#     all_file[i] = all_file[i].replace("\\", "/")
# validate_file = load_json("standard_data.json")["fold_1"]["validate"]
# for i in range(len(validate_file)):
#     validate_file[i] = validate_file[i].replace(
#         "data_new", "ProcessedData/regression_new"
#     )

# train_files = np.setdiff1d(all_file, validate_file)

# column_name = ["SUVmax", "SUVmean", "SUVmin"]
# column_name.extend([str(i) for i in range(1, 1025)])

# total = []
# for file_path in train_files:
#     data = np.load(file_path)
#     hu, seg = data["HU"], data["segmentation"]
#     np.clip(hu, -1000, 1000, out=hu)
#     inp = ((hu + 1000.0) / 2000.0 + seg) * 0.5
#     inp = np.reshape(inp, (-1))
#     tar = np.array([data["max"], data["mean"], data["min"]])
#     result = np.concatenate((tar, inp), axis=0)
#     total.append(result.tolist())

# df = pd.DataFrame(total)
# df.columns = column_name
# df.to_csv("PET-CT.csv", index=False)


# data = pd.read_csv("PET-CT.csv").values
# data = data.astype(np.float32)
# SUVdict = {}
# for i in data:
#     SUVdict[str(i[0])] = i[1:3]

# file_list = [
#     "WERCS-auto-NA-none.csv",
#     "WERCS-auto-0.5-balance.csv",
#     "WERCS-auto-0.5-extreme.csv",
#     "WERCS-auto-0.7-balance.csv",
#     "WERCS-auto-0.7-extreme.csv",
# ]

# for file in file_list:
#     file_data = pd.read_csv(file)
#     file_data_value = file_data.values
#     file_data_value = file_data_value.astype(np.float32)
#     file_data_value = file_data_value[:, 1:]
#     new_value = []
#     for v in file_data_value:
#         new_v = np.insert(v, 1, SUVdict[str(v[0])])
#         new_value.append(new_v)
#     df = pd.DataFrame(new_value)
#     df.to_csv("-" + file, index=False)
# print("Done")


def k_fold_no_validate(K: int, excel: str):
    values = pd.read_csv(excel).values
    result = {
        "0-1": [],
        "1-2": [],
        "2-3": [],
        "3-4": [],
        "4-5": [],
        "5-10": [],
        "10-20": [],
        "20-30": [],
        "30-50": [],
        "50-": [],
    }
    for i, d in enumerate(values):
        max = d[0]
        if 0 <= max < 1:
            result["0-1"].append(i)
        elif 1 <= max < 2:
            result["1-2"].append(i)
        elif 2 <= max < 3:
            result["2-3"].append(i)
        elif 3 <= max < 4:
            result["3-4"].append(i)
        elif 4 <= max < 5:
            result["4-5"].append(i)
        elif 5 <= max < 10:
            result["5-10"].append(i)
        elif 10 <= max < 20:
            result["10-20"].append(i)
        elif 20 <= max < 30:
            result["20-30"].append(i)
        elif 30 <= max < 50:
            result["30-50"].append(i)
        elif 50 <= max:
            result["50-"].append(i)
    avg_split = {}
    for key, value in result.items():
        each_fold_size = len(value) // (K)
        key_files = []
        for i in range(K):
            fold = np.random.choice(result[key], each_fold_size, False)
            result[key] = np.setdiff1d(result[key], fold)
            key_files.append(fold)
        avg_split[key] = key_files
        for i in range(len(result[key])):
            avg_split[key][i] = np.append(avg_split[key][i], result[key][i])
    # 拼接已平均分的数据，分成训练集、测试集和验证集。
    final_spilt = {}
    for i in range(K):
        train_set = []
        test_set = []
        for key, value in avg_split.items():
            for j in range(K):
                if i == j:
                    test_set.extend(value[j])
                else:
                    train_set.extend(value[j])
        final_spilt[f"fold_{i+1}"] = {
            "train": train_set,
            "test": test_set,
        }
    return final_spilt


a = k_fold_no_validate(5, "-WERCS-auto-0.5-balance.csv")

