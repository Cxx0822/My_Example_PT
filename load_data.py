# -*-coding:utf-8-*-
import torch
import torchvision
from torchvision import datasets, transforms
import os
import os.path


def load_data(data_dir, image_size, batch_size):
    data_transform = transforms.Compose([transforms.Resize([image_size, image_size]), transforms.ToTensor()])

    image_datasets = datasets.ImageFolder(root=os.path.join(data_dir), transform=data_transform)

    dataloader = torch.utils.data.DataLoader(dataset=image_datasets, batch_size=batch_size, shuffle=True)

    return dataloader
