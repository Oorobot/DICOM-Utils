from ThreePhaseBone import *
from utils import *


if __name__ == "__main__":

    # csv [ 0 -> bodypart; 1 -> type; 2 -> filename; 3 -> label]
    # type: 3 4, 4 4, 0 0

    files_info = np.loadtxt(
        "ThreePhaseBone/total.csv", dtype=str, delimiter=",", skiprows=1
    )
    dirs = os.listdir("ThreePhaseBone/2015-2021")

    for info, d in zip(files_info, dirs):

        filename = os.path.join("ThreePhaseBone/2015-2021", d, info[2])
        classes = 1 if info[3] == "1" else 0
        result_path = info[0] + "/" + d

        img_process(filename, info[1], classes, result_path)

    print(0)
