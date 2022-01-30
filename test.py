import matplotlib.pyplot as plt
import cv2
import numpy as np

from dicom import (
    dcmread,
    get_pixel_value,
    convert_dicom_to_gary_image_by_window_center_and_window_width,
)

file = "ThreePhaseBone/055/055_FLOW.dcm"

dicom = dcmread(file)
window_center, window_width = float(dicom.WindowCenter), float(dicom.WindowWidth)
pixel_value = get_pixel_value(file)
gray_image = convert_dicom_to_gary_image_by_window_center_and_window_width(
    pixel_value, window_center, window_width, to_uint8=True, ratio=0.5
)
other_method = np.clip(pixel_value, 0, np.max(pixel_value[0:20]) * 0.5) / (
    np.max(pixel_value[0:20]) * 0.5
)
other_method = (other_method * 255).astype(np.uint8)
# flip_other_method = np.flip(other_method, -1)

for i in range(25):
    cv2.imwrite(f"ProcessedData/flow_gray_{i}.png", other_method[i])

