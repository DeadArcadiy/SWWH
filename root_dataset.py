import torch
import albumentations as albu
import cv2
from tqdm import tqdm
import os

class Root_dataset(torch.utils.data.Dataset):
    def __init__(self,augmentations,image_folder,mask_folder,device):
        self.image_folder = image_folder
        self.mask_folder = mask_folder
        self.transforms = augmentations
        self.images = os.listdir(image_folder)
        self.device = device
        print(self.images)

    def __getitem__(self,i):
        image = cv2.imread(self.image_folder+'/'+self.images[i])
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.mask_folder+'/'+self.images[i],cv2.IMREAD_GRAYSCALE)
        transformed = self.transforms(image=image, mask=mask)
        transformed_image = transformed['image']
        transformed_mask = transformed['mask']
        transformed_mask = transformed_mask.reshape(512,512,1)
        transformed_image = torch.Tensor(transformed_image)
        transformed_mask = torch.Tensor(transformed_mask)
        transformed_image = transformed_image.permute((2, 0, 1))
        transformed_mask = transformed_mask.permute((2, 0, 1))
        transformed_image /= 255
        return transformed_image.to(self.device),transformed_mask.to(self.device)
            
            
    def __len__(self):
        return len(self.images)
