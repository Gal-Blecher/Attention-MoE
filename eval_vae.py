from config import setup
import torch
import datasets
import build
import train
import plots
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.models import resnet18
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader

model = torch.load('/Users/galblecher/Desktop/Thesis_out/ssl_cifar/unsupervised only/vae.pkl', map_location=torch.device('cpu'))

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
dataset = CIFAR10(root='path_to_cifar10_data', train=False, transform=transform, download=True)
dataloader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=4)

batch = next(iter(dataloader))
