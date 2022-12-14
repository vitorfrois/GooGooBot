import torch
import torchvision
import torchvision.transforms as transforms 
import torch.nn as nn
import time 
import copy
from PIL import Image

MODEL_PATH = 'model.pth'

classes = ('Plane', 'Car', 'Bird', 'Cat',
           'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck')

def img_tensor(img_path):
    image = Image.open(img_path)
    image = image.resize((32, 32))
    transform = transforms.Compose([
        transforms.PILToTensor(),
        transforms.ConvertImageDtype(torch.float),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])
    ])
    tensor = transform(image)
    tensor = tensor.unsqueeze(0)
    return tensor

def create_net(MODEL_PATH):
    torch.device('cpu')
    net = torchvision.models.resnet18(weights='DEFAULT')
    num_ftrs = net.fc.in_features
    net.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_ftrs, 10)
    )
    net.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    net.eval()
    return net