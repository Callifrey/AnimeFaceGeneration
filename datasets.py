import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
import os
import numpy as np

class AnimeFaceDatasets(Dataset):
    def __init__(self, root=''):
        self.transforms = transforms.Compose([transforms.Resize((64,64)),
                                              transforms.ToTensor(),
                                              transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
        self.root = root
        img_path = os.listdir(root)  # abs path for all images
        self.img_path = [os.path.join(root, k) for k in img_path]


    def __getitem__(self, item):
        image = Image.open(self.img_path[item])
        image = self.transforms(image)
        return image

    def __len__(self):
        return len(self.img_path)

