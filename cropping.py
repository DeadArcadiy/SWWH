import PIL
from matplotlib import pyplot as plt
import pandas
import cv2
import os
import shutil
from tqdm import tqdm
import numpy as np


class ImageCropper:
    def __init__(self, path, save_path):
        self.path = path
        self.save_path = save_path

    def image_maker(self, name):
        image = cv2.imread(self.path + name, 3)
        b, g, r = cv2.split(image)
        image = cv2.merge([r, g, b])
        return image

    @staticmethod
    def rectangle_crop(self, image):
        rectangle_cropped_image = None
        original = image.copy()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

        cnts = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]

        for c in cnts:
            x, y, weight, height = cv2.boundingRect(c)
            if weight > 1000 and height > 1000:
                if abs(weight - height) < 500:
                    cv2.rectangle(image,
                                (x, y),
                                (x + weight, y + height),
                                (36, 255, 12),
                                2)
                    rectangle_cropped_image = original[y:y + height, x:x + weight]
        if rectangle_cropped_image is not None:
            return rectangle_cropped_image
        else:
            return None

    @staticmethod
    def circle_crop(self, rectangle_cropped_image):
        height, diametr = rectangle_cropped_image.shape[:2]
        mask1 = np.zeros_like(rectangle_cropped_image)
        mask1 = cv2.circle(mask1,
                           (height // 2, diametr // 2),
                           int(diametr / 2 - diametr / 20),
                           (255, 255, 255),
                           -1)
        circle_cropped_image = cv2.cvtColor(rectangle_cropped_image, cv2.COLOR_BGR2RGBA)
        circle_cropped_image[:, :, 3] = mask1[:, :, 0]
        return circle_cropped_image
    
    def crop_image(self,name):
        img = ImageCropper.image_maker(self,name)
        recimg = ImageCropper.rectangle_crop(self,img)
        if recimg is not None:
            res =  ImageCropper.circle_crop(self,recimg)
            cv2.imwrite(self.save_path+name[:-3]+'png',res)
        else:
            print('Bad image name:' + name)
