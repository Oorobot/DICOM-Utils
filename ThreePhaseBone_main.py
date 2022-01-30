from datetime import datetime
from SimpleITK.extra import GetArrayFromImage
from glob import glob

from ThreePhaseBone import *
from utils import *


# 校验DCM和xlsx之间病患信息是否匹配
# flows = glob("ThreePhaseBone/*/*FLOW.dcm")
# validate_dcm(flows, "ThreePhaseBone/2021-11-12.xlsx")

# 数据处理

# xlsx = pd.read_excel("ThreePhaseBone/2021-11-12.xlsx")
# infos = xlsx[["编号", "最终结果", "部位", "type"]].values
# JPG数据处理
# jpgs = glob("ThreePhaseBone/*/*_1.JPG")
# for jpg in jpgs:
#     index = jpg.split("\\")[-1].split("_")[0]
#     info = infos[int(index) - 1]
#     bodypart = "hip" if info[2] == "髋" else "knee"
#     save_path = os.path.join("ProcessedData", bodypart, index)
#     img_process(jpg, info[-1], info[1], save_path)

# DCM数据处理
# dcms = glob("ThreePhaseBone/*/*_FLOW.dcm")
# for dcm in dcms:
#     index = dcm.split("\\")[-1].split("_")[0]
#     info = infos[int(index) - 1]
#     bodypart = "hip" if info[2] == "髋" else "knee"
#     save_path = os.path.join("ProcessedData", bodypart, index)
#     dcm_process(dcm, info[1], save_path)

# DCMnpzs = glob("ProcessedData/*/*DCM.npz")
# FPtxt = open("TPB.txt", "w+")
# for npz in DCMnpzs:
#     print("filename: ", npz, file=FPtxt)
#     npz = np.load(npz)
#     print(
#         "the max of flow: %d, the min: %d."
#         % (np.max(npz["data"][0:20]), np.min(npz["data"][0:20])),
#         file=FPtxt,
#     )
#     print(
#         "the max of pool: %d, the min: %d."
#         % (np.max(npz["data"][20:]), np.min(npz["data"][20:])),
#         file=FPtxt,
#     )

print("Done.")
