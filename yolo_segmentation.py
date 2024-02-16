from ultralytics import YOLO
import torch
# Load a model
if __name__ == '__main__':
    print(torch.cuda.is_available())
    model = YOLO("yolov8n.yaml")  # build a new model from scratch
    model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)
    model.eval()
# Use the model
    model.train(data='train.yaml', epochs=50, batch=4, workers=4, patience=10)  # train the model
    metrics = model.val()  # evaluate model performance on the validation set
