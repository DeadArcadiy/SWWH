from cropping import ImageCropper
import PIL
from matplotlib import pyplot as plt
import pandas
import cv2
import os
import shutil
from tqdm import tqdm
import numpy as np


source_dir = 'arabidopsis/'
savepath = 'cropped/'
cropper = ImageCropper(path = source_dir,save_path=savepath)
for file_name in tqdm(os.listdir(source_dir)):
    cropper.crop_image(file_name)


