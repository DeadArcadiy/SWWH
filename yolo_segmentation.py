from ultralytics import YOLO
import torch
# Load a model
if __name__ == '__main__':
    print(torch.cuda.is_available())
    model = YOLO('yolov8n-seg.yaml')  # build a new model from YAML
    model = YOLO('yolov8n-seg.pt')  # load a pretrained model (recommended for training)
    model = YOLO('yolov8n-seg.yaml').load('yolov8n.pt')  # build from YAML and transfer weights

    model.train(data='train.yaml', epochs=100, batch=4, patience=10)  # train the model
    #metrics = model.val()  # evaluate model performance on the validation set
