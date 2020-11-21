from datasets import *
import torch
import torch.utils.data.dataloader as dataloader
import matplotlib.pyplot as plt

def img_show(imgs):
    plt.figure()
    for i, img in enumerate(imgs):
        img = img.permute(1,2,0)
        ax = plt.subplot(4,4,i+1)
        plt.axis('off')
        plt.imshow(img)
    plt.show()

root = './data/data'
anime_dataset = AnimeFaceDatasets(root)
anime_dataloader = dataloader.DataLoader(anime_dataset, batch_size=16, shuffle=False)

data = iter(anime_dataloader)
images = next(data)

img_show(images)

