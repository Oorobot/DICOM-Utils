from glob import glob
import math

from PETCT import *
from utils import *

# # 分割标签数据校验
# lung_slice = np.loadtxt("lung_slice.csv", np.uint32, delimiter=",", usecols=(1, 2))
# seg_files = glob("PET-CT/*/*.nii.gz")

# for file in seg_files:
#     idx = int(file.split("\\")[-1].split(".")[0])
#     start, end = lung_slice[idx - 1]
#     img = sitk.ReadImage(file)
#     length = img.GetSize()[-1]
#     if length != end + 1 - start:
#         print("not mathch, file: ", file)

# # reg数据校验
reg_data = glob("ProcessedData/regression/*.npz")
# regression_data_validate(reg_data, "valid.txt")

# regession 数据 按找 suvmax 进行划分, 画 suvmax, suvmin, suvmean 直方图
suvmax = []
suvmin = []
suvmean = []
max0_1 = []
max1_2 = []
max2_3 = []
max3_4 = []
max4_5 = []
max5_10 = []
max10_ = []
for file in reg_data:
    data = np.load(file)
    max = data["max"]
    mean = data["mean"]
    min = data["min"]
    suvmax.append(max)
    suvmean.append(mean)
    suvmin.append(min)
    if 0 <= max < 1:
        max0_1.append(file)
    elif 1 <= max < 2:
        max1_2.append(file)
    elif 2 <= max < 3:
        max2_3.append(file)
    elif 3 <= max < 4:
        max3_4.append(file)
    elif 4 <= max < 5:
        max4_5.append(file)
    elif 5 <= max < 10:
        max5_10.append(file)
    elif max >= 10:
        max10_.append(file)

suvmax = np.array(suvmax)
suvmin = np.array(suvmin)
suvmean = np.array(suvmean)


def hist(array: np.ndarray, save_path: str):

    plt.figure()
    plt.hist(
        array, math.ceil(np.max(array) - np.min(array)), (np.min(array), np.max(array)),
    )
    plt.savefig(save_path)
    plt.close()


hist(suvmax, "suvmax.png")
hist(suvmin, "suvmin.png")
hist(suvmean, "suvmean.png")

save_json(
    "ProcessedData/regression/data.json",
    {
        "0-1": max0_1,
        "1-2": max1_2,
        "2-3": max2_3,
        "3-4": max3_4,
        "4-5": max4_5,
        "5-10": max5_10,
        "10-": max10_,
    },
)

