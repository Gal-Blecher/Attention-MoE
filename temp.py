import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torch.nn.functional as F
import random
from torch.utils.data import DataLoader, SubsetRandomSampler
import nets


def label_samples(model, unlabeled_trainloader, labeled_trainloader, th=0.5):
    model.eval()
    rm_lst = []
    with torch.no_grad():
        idx = []
        for batch_idx, (inputs, target) in enumerate(unlabeled_trainloader):
            batch_indices = unlabeled_trainloader.sampler.indices[batch_idx * unlabeled_trainloader.batch_size: (batch_idx + 1) * unlabeled_trainloader.batch_size]
            inputs = inputs.to(device)
            outputs = model(inputs)
            outputs = F.softmax(outputs, 1)
            scores, predictions = torch.max(outputs, dim=1)
            indices = unlabeled_trainloader.sampler.indices

            # use the inputs, labels, scores, and predictions as needed
            for i in range(inputs.size(0)):
                if scores[i] >= th:
                    labeled_trainloader.sampler.indices.append(batch_indices[i])
                    rm_lst.append(batch_indices[i])
    # unlabeled_trainloader.sampler.indices = [set(unlabeled_trainloader.sampler.indices) - set(rm_lst)]
    unlabeled_trainloader.sampler.indices = list(set(unlabeled_trainloader.sampler.indices) - set(rm_lst))


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# Load the CIFAR10 dataset and select the first 1000 labeled examples
trainset_full = torchvision.datasets.CIFAR10(root='./data', train=True,
                                             download=True, transform=transform_train)
labeled_indexes = random.sample(range(trainset_full.data.shape[0]), 1000)
# Select the remaining 49000 unlabeled examples
unlabeled_indexes = list(set(range(len(trainset_full))) - set(labeled_indexes))




model = nets.SimpleModel()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
criterion = nn.CrossEntropyLoss()

for epoch in range(50):
    labeled_sampler = SubsetRandomSampler(labeled_indexes)
    labeled_trainloader = torch.utils.data.DataLoader(trainset_full, batch_size=64, sampler=labeled_sampler)

    unlabeled_sampler = SubsetRandomSampler(unlabeled_indexes)
    unlabeled_trainloader = torch.utils.data.DataLoader(trainset_full, batch_size=64, sampler=unlabeled_sampler, shuffle=False)


    model.train()
    running_loss = 0.0
    for i, data in enumerate(labeled_trainloader, 0):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 5 == 0:
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0
    label_samples(model, unlabeled_trainloader, labeled_trainloader, th=0.3)
    labeled_indexes = labeled_trainloader.sampler.indices
    unlabeled_indexes = unlabeled_trainloader.sampler.indices




