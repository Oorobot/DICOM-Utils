import argparse
import json
import os
from glob import glob

import cv2
import matplotlib
import numpy as np
import SimpleITK as sitk

from utils.dicom import ct2image, suvbw2image
from utils.html import HTML

RESULT_FILES = "./Files/FRI/result_files"
RESULT_IMAGES = "./Files/FRI/result_images"
GT2COLOR = {
    "infected": (0, 0, 1),  # 红色
    "uninfected": (0, 1, 0),  # 绿色
    "bladder": (1, 0, 0),  # 蓝色
    "lesion": (0, 0, 1),  # 红色
}
DR2COLOR = {
    "infected": (1, 0, 1),
    "uninfected": (0, 1, 1),
    "bladder": (1, 1, 0),
    "lesion": (63 / 255, 133 / 255, 205 / 255),  # 蓝品红
}


def draw_label(img, b, is_gt: bool):
    label_color = GT2COLOR[b[0]] if is_gt else DR2COLOR[b[0]]
    cv2.rectangle(img, b[1:3], b[3:5], color=label_color, thickness=1)
    labelSize = cv2.getTextSize(b[0], cv2.FONT_HERSHEY_SIMPLEX, 0.3, 1)[0]
    org = (b[1], b[2] - 1) if is_gt else (b[1], b[4] + labelSize[1] + 1)
    cv2.putText(
        img,
        b[0],
        org,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.3,
        color=label_color,
        thickness=1,
    )


def save_3d_label(no, gts, dts):
    object_image = sitk.ReadImage(
        os.path.join("Files/FRI/image_2mm", f"{no}_Label_Object.nii.gz")
    )
    object_array = sitk.GetArrayFromImage(object_image)
    dt_array = np.zeros_like(object_array)
    image_shape = dt_array.shape[::-1]
    for gt in gts:
        for i in range(3):
            gt[i + 4] -= 1
        c, x1, y1, z1, x2, y2, z2 = gt
        if c == 4:
            continue
        xy = np.full((y2 - y1, x2 - x1), c)
        dt_array[z1, y1:y2, x1:x2] = xy
        dt_array[z2, y1:y2, x1:x2] = xy
        xz = np.full((z2 - z1, x2 - x1), c)
        dt_array[z1:z2, y1, x1:x2] = xz
        dt_array[z1:z2, y2, x1:x2] = xz
        yz = np.full((z2 - z1, y2 - y1), c)
        dt_array[z1:z2, y1:y2, x1] = yz
        dt_array[z1:z2, y1:y2, x2] = yz
    for dt in dts:
        for i in range(3):
            dt[i + 1] = max(dt[i + 1], 0)
            dt[i + 4] = min(dt[i + 4], image_shape[i]) - 1
        c, x1, y1, z1, x2, y2, z2 = dt
        xy = np.full((y2 - y1, x2 - x1), c + 5)
        dt_array[z1, y1:y2, x1:x2] = xy
        dt_array[z2, y1:y2, x1:x2] = xy
        xz = np.full((z2 - z1, x2 - x1), c + 5)
        dt_array[z1:z2, y1, x1:x2] = xz
        dt_array[z1:z2, y2, x1:x2] = xz
        yz = np.full((z2 - z1, y2 - y1), c + 5)
        dt_array[z1:z2, y1:y2, x1] = yz
        dt_array[z1:z2, y1:y2, x2] = yz
    # 保存.nii.gz图像
    dt_image = sitk.GetImageFromArray(dt_array)
    sitk.WriteImage(dt_image, f"Files/FRI/dts/{no}.nii.gz")


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--op", type=str, choices=['json', 'html'], default='json')
    parser.add_argument("--result-file", type=str, default=None)
    parser.add_argument("--preffixes", type=str, nargs="+", default=None)
    parser.add_argument("--html-name", type=str, default="index")
    # 01_b6_n1_6
    # 01_b6_n1_6_resnet18_d_pet
    # 01_b6_n1_6_resnet18_d_pet_mip_mixup0.2_result
    # 01_b6_n1_6__f2_el1
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = args_parser()
    if args.op == "json":
        if args.result_file is None:
            raise Exception("请输入result file.")

        dectetions = json.load(open(args.result_file))
        preffix = os.path.splitext(os.path.basename(args.result_file))[0]
        print(preffix)

        gray_cmap = matplotlib.colormaps["gray"]
        gray_colors = gray_cmap(np.linspace(0, 1, gray_cmap.N))[:, :3]
        hot_cmap = matplotlib.colormaps["hot"]
        hot_colors = hot_cmap(np.linspace(0, 1, hot_cmap.N))[:, :3]

        ground_truth: dict = dectetions["ground_truth"]
        detection_results: dict = dectetions["detection_results"]
        print("start...")
        for no, item in ground_truth.items():
            class_boxes = {
                "infected": [],
                "uninfected": [],
                "bladder": [],
                "lesion": [],
            }
            for _ in item:
                if _["class_name"] in class_boxes:
                    class_boxes[_["class_name"]].append(
                        ["GT", _["class_name"]] + _["bbox"]
                    )
            for _ in detection_results[no]:
                if _["class_name"] in class_boxes:
                    class_boxes[_["class_name"]].append(
                        ["DR", _["class_name"]] + _["bbox"]
                    )

            lesion_boxes = (
                class_boxes["lesion"]
                + class_boxes["infected"]
                + class_boxes["uninfected"]
            )
            # 按照Y轴进行排序
            result_images = []
            lesion_boxes.sort(key=lambda x: x[3])
            image = None
            for bbox in lesion_boxes:
                t, c, x1, y1, z1, x2, y2, z2 = bbox
                if image is None:
                    image = [y1, y2, [[c, x1, z1, x2, z2, t]]]
                else:
                    ma = max(image[0], y1)
                    mi = min(image[1], y2)
                    if ma < mi:
                        image[0] = ma
                        image[1] = mi
                        image[2].append([c, x1, z1, x2, z2, t])
                    else:
                        result_images.append(image)
                        image = [y1, y2, [[c, x1, z1, x2, z2, t]]]
            if image is not None:
                result_images.append(image)

            # 读取 CT 和 SUV
            ct_image = sitk.ReadImage(
                os.path.join("Files/FRI/image_2mm", f"{no}_CT.nii.gz")
            )
            suv_image = sitk.ReadImage(
                os.path.join("Files/FRI/image_2mm", f"{no}_SUVbw.nii.gz")
            )
            ct_array = sitk.GetArrayFromImage(ct_image)
            suv_array = sitk.GetArrayFromImage(suv_image)
            hu = ct2image(ct_array, 300, 1500, True)
            suvbw = suvbw2image(suv_array, 2.5, True)

            for i, bbox in enumerate(result_images):
                y = (int)((bbox[0] + bbox[1]) / 2)
                hu_slice = hu[:, y, :]
                suvbw_slice = suvbw[:, y, :]

                hu_image = gray_colors[hu_slice]
                suv_image = gray_colors[255 - suvbw_slice]
                hot_suv_image = hot_colors[suvbw_slice]
                # BGR -> RGB
                suv_image = suv_image[:, :, ::-1]
                hot_suv_image = hot_suv_image[:, :, ::-1]
                # 翻转Z轴
                petct = cv2.addWeighted(hu_image, 0.7, hot_suv_image, 0.3, 0)
                hu_image = np.ascontiguousarray(np.flip(hu_image, 0))
                suv_image = np.ascontiguousarray(np.flip(suv_image, 0))
                petct = np.ascontiguousarray(np.flip(petct, 0))
                # 画图
                D = hu_image.shape[0] - 1
                for b in bbox[2]:
                    if b[0] == "other":
                        continue
                    b[4], b[2] = D - b[2], D - b[4]
                    draw_label(hu_image, b, b[-1] == "GT")
                    draw_label(suv_image, b, b[-1] == "GT")
                    draw_label(petct, b, b[-1] == "GT")
                cv2.imwrite(
                    os.path.join(RESULT_IMAGES, f"{preffix}-{no}-{i}.png"),
                    np.hstack([hu_image, suv_image, petct]) * 255,
                )
        print("end.")
    if args.op == "html":
        if args.preffixes is None:
            raise Exception("请输入preffixes")
        html = HTML("骨折相关感染", RESULT_IMAGES, file_name=args.html_name)
        dectetions = json.load(
            open(os.path.join("Files/FRI/result_files", f"{args.preffixes[0]}.json"))
        )
        nos = list(dectetions["ground_truth"].keys())
        nos.sort()
        for no in nos:
            html.add_header(no)
            imgs = []
            for d in args.preffixes:
                imgs += glob(os.path.join(RESULT_IMAGES, f"{d}-{no}-*"))
            imgs.sort()
            titles = []
            ims = []
            for img in imgs:
                img_name = os.path.basename(img)
                ims.append(img_name)
                titles.append(img_name.split(".")[0])
            html.add_images(ims, titles)
        html.save()
