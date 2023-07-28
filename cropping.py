import PIL
from matplotlib import pyplot as plt
import pandas
import cv2
import os
import shutil
from tqdm import tqdm
import numpy as np


class ImageCropper:
    def __init__(self, path, name):
        #        self.path = path
        #        self.name = name
        self.image = cv2.imread(path + name + ".JPG")

    def rectangle_crop(self):
        rectangle_cropped_image = None
        original = self.image.copy()
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

        cnts = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]

        for c in cnts:
            x, y, weight, height = cv2.boundingRect(c)
            if weight > 500 and height > 500:
                cv2.rectangle(self.image,
                              (x, y),
                              (x + weight, y + height),
                              (36, 255, 12),
                              2)
                ROI = original[y:y + height, x:x + weight]
                rectangle_cropped_image = ROI.copy()
        if rectangle_cropped_image is not None:
            return rectangle_cropped_image
        else:
            raise Exception

    @staticmethod
    def circle_crop(self, rectangle_cropped_image):
        img = rectangle_cropped_image.copy()
        height, diametr = img.shape[:2]
        mask1 = np.zeros_like(img)
        mask1 = cv2.circle(mask1,
                           (height // 2, diametr // 2),
                           int(diametr / 2 - diametr / 20),
                           (255, 255, 255),
                           -1)
        circle_cropped_image = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        circle_cropped_image[:, :, 3] = mask1[:, :, 0]
        return circle_cropped_image
