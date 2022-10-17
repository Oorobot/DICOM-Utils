import os
import shutil
from glob import glob

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydicom
import SimpleITK as sitk
from mlxtend.evaluate import mcnemar, mcnemar_table  # 用于计算显著性水平 p

from utils.dicom import get_patient_info, get_SUVbw_in_GE, read_serises_image
from utils.metric import classification_metrics
from utils.utils import load_json, mkdir, save_json, to_pinyin
