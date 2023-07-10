import json
import os
import zipfile
from glob import glob

import cv2
import numpy as np
import pandas as pd
import SimpleITK as sitk

from utils.dicom import (
    data_preprocess,
    get_patient_infomation,
    resameple_based_size,
    suvbw2image,
)
from utils.utils import LABELS, mixup, random_flip, ricap


def main():
    # excel
    excel = pd.read_excel("Files/Data/PETCT-FRI/PET-FRI.xlsx", sheet_name="FRI-PETall")
    # 验证集上的病人编号
    val_nos = json.load(open("Files/FRI/dataset.json"))['fold 1']['val']
    # 纳入研究的所有病人编号
    total_nos = list(LABELS.keys())

    total_nos_int = [int(no) for no in total_nos]
    total_patiens = excel.query("No_FRI in @total_nos_int")
    # 统计所有病人的年龄分布
    ages = total_patiens[["Age"]].values
    print(f"Age: {np.mean(ages):.1f}({np.min(ages)}-{np.max(ages)})")

    # 统计所有病人的性别分布
    sex_num = total_patiens.value_counts(subset=["Gender"])
    print(sex_num)

    # 统计所有病人的 body mass index
    total_bmi = []
    for no in total_nos:
        files = glob(f"Files/Data/PETCT-FRI/NormalData/{no}/PET/*")
        information = get_patient_infomation(files[0])
        bmi = information["Patient Weight"] / (information["Patient Size"] ** 2)
        total_bmi.append(bmi)
    print(
        f"BMI: {np.mean(total_bmi):.1f}({np.min(total_bmi):.1f}-{np.max(total_bmi):.1f})"
    )

    # 导出验证集中病人的PETCT影像文件
    with zipfile.ZipFile('Files/FRI/Experiments/val.zip', 'w') as zip:
        for no in val_nos:
            for dir_name in ['CT', 'PET']:
                current_dir = os.path.join(
                    "Files", "Data", "PETCT-FRI", "NormalData", no, dir_name
                )
                zip_dir = os.path.join("NormalData", no, dir_name)
                for filename in os.listdir(current_dir):
                    zip.write(
                        os.path.join(current_dir, filename),
                        os.path.join(zip_dir, filename),
                    )

    # 导出验证集上的病人编号、名字、检查日期
    val_nos_int = [int(no) for no in val_nos]
    patiens = excel.query("No_FRI in @val_nos_int")
    information = patiens[["No_FRI", "Name", "Date"]]
    information.to_excel("Files/FRI/Experiments/information.xlsx")


def clip(dirname):
    images = glob(f"{dirname}/*.jpg")
    for image in images:
        img = cv2.imread(image)
        img = img[220:1560, 1030:1935, :]
        cv2.imwrite(os.path.basename(image), img, [cv2.IMWRITE_JPEG_QUALITY, 100])


def visualize_data_augmentation(no1: str, no2: str):
    def write(image, boxes, name):
        with open(f"boxes-{name}.json", "w") as f:
            json.dump(boxes.tolist(), f)
        ct_image = sitk.GetImageFromArray(image[0, ...])
        pet_image = sitk.GetImageFromArray(image[1, ...])
        sitk.WriteImage(ct_image, f"ct-{name}.nii.gz")
        sitk.WriteImage(pet_image, f"pet-{name}.nii.gz")

    _, image1, box1 = data_preprocess(
        no1,
        input_type='petct',
        input_shape=[384, 96, 160],
        to_label={'infected': 0, 'uninfected': 1, 'bladder': 2},
    )
    write(image1, box1, no1)

    _, image2, box2 = data_preprocess(
        no2,
        input_type='petct',
        input_shape=[384, 96, 160],
        to_label={'infected': 0, 'uninfected': 1, 'bladder': 2},
    )
    write(image2, box2, no2)

    # ricap
    image_1, box_1 = ricap(image1, box1, image2, box2, 1)
    image_2, box_2 = ricap(image1, box1, image2, box2, 3)
    write(image_1, box_1, f"{no1}-{no2}-ricap1")
    write(image_2, box_2, f"{no1}-{no2}-ricap3")

    # mixup
    image, box = mixup(image1, box1, image2, box2)
    write(image, box, f'{no1}-{no2}-mixup')

    # flip
    image1, box1 = random_flip(image1, box1, 0)
    write(image1, box1, no1 + '-flip1')
    image1, box1 = random_flip(image1, box1, 0)
    image1, box1 = random_flip(image1, box1, 1)
    write(image1, box1, no1 + '-flip2')
    image1, box1 = random_flip(image1, box1, 1)
    image1, box1 = random_flip(image1, box1, 2)
    write(image1, box1, no1 + '-flip3')


def visualize_cropped_lesion_and_pet_mip():
    ct = sitk.ReadImage("ct-143.nii.gz")
    pet = sitk.ReadImage("pet-143.nii.gz")
    ct_array = sitk.GetArrayFromImage(ct)
    pet_array = sitk.GetArrayFromImage(pet)
    boxes = json.load(open("boxes-143.json"))
    for i, box in enumerate(boxes):
        if box[-1] == 0 or box[-1] == 1:
            lesion_ct = ct_array[
                box[2] : box[5] + 1, box[1] : box[4] + 1, box[0] : box[3] + 1
            ]
            lesion_pet = pet_array[
                box[2] : box[5] + 1, box[1] : box[4] + 1, box[0] : box[3] + 1
            ]
            lesion_ct = sitk.GetImageFromArray(lesion_ct)
            lesion_pet = sitk.GetImageFromArray(lesion_pet)
            sitk.WriteImage(lesion_ct, f"ct-143-lesion-{i}.nii.gz")
            sitk.WriteImage(lesion_pet, f"pet-143-lesion-{i}.nii.gz")
            lesion_pet_resampled = resameple_based_size(lesion_pet, (64, 64, 64))
            pet = sitk.GetArrayFromImage(lesion_pet_resampled)
            pet_xy = suvbw2image(np.max(pet, axis=0), 2.5, True)
            pet_xz = suvbw2image(np.max(pet, axis=1), 2.5, True)
            pet_yz = suvbw2image(np.max(pet, axis=2), 2.5, True)
            cv2.imwrite(f"ct-143-lesion-{i}-xy.jpg", pet_xy)
            cv2.imwrite(f"ct-143-lesion-{i}-xz.jpg", pet_xz)
            cv2.imwrite(f"ct-143-lesion-{i}-yz.jpg", pet_yz)


if __name__ == "__main__":
    main()
