import os
import albumentations as albu
import tensorflow as tf
import numpy as np

class DatasetCreator:
   def __init__(self,preprocess) -> None:
      self.preprocess = preprocess

   def get_mask(self,image):
      return tf.strings.split(image,os.path.sep)[-1]

   def process_image_with_mask(self,file_path):
      image = tf.io.read_file(file_path)
      image = tf.image.decode_png(image)
      mask = self.maskspath + self.get_mask(file_path)
      mask = tf.io.read_file(mask)
      mask = tf.image.decode_png(mask)
      cond = tf.greater_equal(mask,1)
      mask = tf.where(cond,1,0)
      image = tf.cast(image,tf.uint8)
      mask = tf.cast(mask,tf.uint8)
      return image,mask
   
   def aug_fn(self,image, mask):
      image = image[:,:,:3]
      data = {"image":image,"mask":mask}
      aug_data = self.transforms(**data)
      image = aug_data["image"]
      mask = aug_data["mask"]
      image = tf.cast(image, tf.float32)
      mask = tf.cast(mask, tf.float32)
      image = self.preprocess(image)
      return image,mask
   
   def process_data(self,image, mask):
    image,mask = tf.numpy_function(self.aug_fn,inp = (image,mask),Tout=(tf.float32,tf.float32))
    return image, mask

   def __call__(self,imagepath,maskspath,transforms):
      self.transforms = transforms
      self.maskspath = maskspath
      dataset = tf.data.Dataset.list_files(imagepath)
      dataset = dataset.map(self.process_image_with_mask)
      dataset = dataset.map(self.process_data)
      dataset = dataset.shuffle(100)
      return dataset.batch(4)