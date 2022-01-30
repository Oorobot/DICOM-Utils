import SimpleITK as sitk
from pydicom import dcmread
import numpy as np


def get_pixel_array(filename: str):
    return dcmread(filename).pixel_array


# get_pixel_array(file) * RescaleSlope + RescaleIntercept = get_pixel_value(file)


def get_pixel_value(filename: str):
    return sitk.GetArrayFromImage(sitk.ReadImage(filename))


def convert_dicom_to_gary_image_by_window_center_and_window_width(
    pixel_value: np.ndarray,
    window_center: float,
    window_width: float,
    ratio=1.0,
    to_uint8=False,
):
    if ratio != 1.0:
        window_width *= ratio
        window_center *= ratio

    min_window = window_center - window_width * 0.5
    gray_image = (pixel_value - min_window) / float(window_width)
    np.clip(gray_image, 0, 1, out=gray_image)
    if to_uint8:
        gray_image = (gray_image * 255).astype(np.uint8)
    return gray_image
