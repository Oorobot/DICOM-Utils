import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
import cv2

# 处理骨三相
def img_process(filename: str, clip_type: str, classes: int, result_path: str):

    # 读取图像(灰度图)
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    # 共裁剪出16张图象, blood flow phase 12张, blood pool phase 4张
    flow_phase = np.zeros((12, 256, 256))
    pool_phase = np.zeros((4, 256, 256))
    cliped_img = []

    #
    if clip_type == "3 4":
        x = 130
        y = 77
        pic_width = 160
        pic_height = 120
        displacement_x = 406
        displacement_y = 169
    elif clip_type == "4 4":
        x = 155
        y = 86 + 125
        pic_width = 110
        pic_height = 86
        displacement_x = 405
        displacement_y = 125

    for i in range(3):
        for j in range(4):
            an_img = img[
                y + i * displacement_y : y + pic_height + i * displacement_y,
                x + j * displacement_x : x + pic_width + j * displacement_x,
            ]
            # resize -> 256x256
            cliped_img.append(an_img)
            an_img = cv2.resize(an_img, (256, 256))
            flow_phase[i * 4 + j] = an_img

    x = 294
    y = 602
    pic_width = 240
    pic_height = 180
    displacement_x = 813
    displacement_y = 253

    for i in range(2):
        for j in range(2):
            an_img = img[
                y + i * displacement_y : y + pic_height + i * displacement_y,
                x + j * displacement_x : x + pic_width + j * displacement_x,
            ]
            # resize -> 256x256
            cliped_img.append(an_img)
            an_img = cv2.resize(an_img, (256, 256))
            pool_phase[i * 2 + j] = an_img

    # 保存文件和裁剪的图片
    for i in range(16):
        plt.subplot(4, 4, i + 1)
        plt.imshow(cliped_img[i], plt.cm.gray)
        plt.axis("off")
    plt.savefig(result_path + ".png")
    plt.close()
    np.savez(result_path + ".npz", flow=flow_phase, pool=pool_phase, label=classes)


def dcm_process(filename: str):
    img = sitk.ReadImage(filename)
    img_array = sitk.GetArrayFromImage(img)
    return img_array

