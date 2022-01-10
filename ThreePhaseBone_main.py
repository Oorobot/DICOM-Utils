from datetime import datetime
from SimpleITK.extra import GetArrayFromImage
from glob import glob

import numpy
from ThreePhaseBone import *
from utils import *
import pydicom
import pandas
from xpinyin import Pinyin
import shutil
import stat

# # 读取数据
# csv [ 0 -> bodypart; 1 -> type; 2 -> filename; 3 -> label]
# type: 3 4, 4 4, 0 0

# files_info = np.loadtxt(
#     "ThreePhaseBone/total.csv", dtype=str, delimiter=",", skiprows=1
# )
# dirs = os.listdir("ThreePhaseBone/2015-2021")

# for info, d in zip(files_info, dirs):

#     filename = os.path.join("ThreePhaseBone/2015-2021", d, info[2])
#     classes = 1 if info[3] == "1" else 0
#     result_path = info[0] + "/" + d

#     img_process(filename, info[1], classes, result_path)

# # 查看 *flow.dcm 文件，每一张图片
# image = sitk.ReadImage("004_FLOW.dcm")
# array = GetArrayFromImage(image)
# print(array.shape)
# plt.figure(dpi=300)
# for i in range(25):
#     plt.subplot(5, 5, i + 1)
#     plt.imshow(array[i] / 255.0, cmap="binary")
#     plt.axis("off")
# plt.savefig(f"ProcessedData/TPB1.png")
# plt.close()

# # 校验拿来的数据是否有误
# flows = glob("TPB/*FLOW.dcm")
# xlsx = pandas.read_excel(
#     "ThreePhaseBone\骨三相-2021-11-2(1).xlsx", sheet_name="Sheet1", usecols=[0, 1, 2]
# )
# p = Pinyin()
# values = xlsx.values
# for flow in flows:
#     img = pydicom.read_file(flow)
#     index = flow.split("\\")[1].split("_")[0]
#     AcqusitionDate = img.AcquisitionDate
#     PatientName = img.PatientName.family_name
#     PatientName = PatientName.replace(" ", "")

#     xlsx_value = values[int(index) - 1]
#     xlsx_name = xlsx_value[2].strip("\t")
#     xlsx_date = xlsx_value[1].strip("\t")
#     xlsx_date = datetime.strptime(xlsx_date, "%Y-%m-%d").strftime("%Y%m%d")

#     xlsx_pinyin = p.get_pinyin(xlsx_name, splitter="", convert="upper")
#     if xlsx_pinyin != PatientName or AcqusitionDate != xlsx_date:
#         print(flow, xlsx_name, xlsx_pinyin, PatientName, xlsx_date, AcqusitionDate)

# 删除ThreePhaseBone原来的dcm文件，注：已有相应的jpg文件，dcm文件多余。
# dcms = glob("ThreePhaseBone/*/*dcm")
# for dcm in dcms:
#     os.chmod(dcm, stat.S_IWRITE)
#     os.remove(dcm)

# 将新的dcm文件移入 ThreeBonePhase 文件夹中
dcms = glob("TPB/*dcm")
for dcm in dcms:
    file_name = dcm.split("\\")[1]
    dir_name = file_name.split("_")[0]
    new_file_path = os.path.join("ThreePhaseBone", dir_name, file_name)
    os.rename(dcm, new_file_path)
# print(0)
