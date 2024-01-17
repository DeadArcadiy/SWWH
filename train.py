from matplotlib import pyplot as plt
from tqdm import tqdm
import torch
from torchmodel import UNet
from root_dataset import Root_dataset
import albumentations as albu
import segmentation_models_pytorch as smp

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    def get_train_augmentation(height):
        trainaugmentation = albu.Compose([
            albu.Resize(height,height),
            albu.HorizontalFlip(p=0.5),
            albu.VerticalFlip(p=0.5),
            albu.RandomRotate90(p=0.5),
            albu.Transpose(p=0.5),
            albu.ShiftScaleRotate(shift_limit=0.01, scale_limit=0.04, rotate_limit=15, p=0.5),
            albu.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.5),
            albu.GridDistortion(p=0.5),
            albu.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
            albu.RandomBrightnessContrast(p=0.5),
            albu.RandomGamma(p=0.5),
            albu.GaussianBlur(blur_limit=(3, 7), p=0.5),
            albu.GaussNoise(),
            albu.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), always_apply=False, p=0.5), 
            #albu.Lambda(image=lambda x,**kwargs: x/255),
        ])
        return trainaugmentation

    def get_val_augmentation(height):
        trainaugmentation = albu.Compose([
            albu.Resize(height,height),
            # albu.Lambda(image=lambda x,**kwargs: x/255),
        ])
        return trainaugmentation

    train_aug = get_train_augmentation(512)
    val_aug = get_val_augmentation(512)

    train = Root_dataset(train_aug,'crosval/train','castom-masks/masks_machine',device)
    val = Root_dataset(val_aug,'crosval/val','castom-masks/masks_machine',device)

    model = UNet(3,1)

    train_loader = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True, num_workers=4)
    val_loader = torch.utils.data.DataLoader(val, batch_size=4, shuffle=False, num_workers=4)

    loss_f = smp.losses.FocalLoss('binary')
    optimizer = torch.optim.Adam(model.parameters(),0.001,amsgrad=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',patience=2)

    epochs = 20

    model.to(device)
    best_loss = 100
    for epoch in range(epochs):
        train_loss = 0.0
        model.train()
        for inputs, targets in tqdm(train_loader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_f(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        train_loss /= len(train_loader)
        val_loss = 0.0

        model.eval()
        for inputs, targets in tqdm(val_loader):
            model.eval()
            outputs = model(inputs)
                
                # Calculate loss
            loss = loss_f(outputs, targets)
            val_loss += loss.item()

        val_loss /= len(val_loader)

        scheduler.step(val_loss)

        if val_loss < best_loss: 
            best_loss = val_loss
            torch.save(model, 'best-model.pt')

        print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

