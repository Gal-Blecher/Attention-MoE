import torch
import torch.nn.functional as F
import random
from torch.utils.data import DataLoader, SubsetRandomSampler
from config import setup
import train
from torch.utils.data import TensorDataset
import numpy as np

def initial_split(train_loader):
    labeled_indexes = random.sample(range(train_loader.dataset.data.shape[0]), setup['ssl'])
    unlabeled_indexes = list(set(range(train_loader.dataset.data.shape[0])) - set(labeled_indexes))
    return labeled_indexes, unlabeled_indexes

def label_samples(model, unlabeled_trainloader, labeled_trainloader, th=0.5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    labeled_data = labeled_trainloader.dataset.tensors[0]
    labeled_labels = labeled_trainloader.dataset.tensors[1]
    unlabeled_dataset = unlabeled_trainloader.dataset
    unlabled_data = torch.tensor([])
    with torch.no_grad():
        for i, input in enumerate(unlabeled_dataset):
            input = input[0].to(device)
            input = input.unsqueeze(0)
            output = model(input)[0]
            output = F.softmax(output, 1)
            score, prediction = torch.max(output, dim=1)

            if score >= th:
                labeled_data = torch.cat([labeled_data, input], dim=0)
                labeled_labels = torch.cat([labeled_labels, prediction], dim=0)
            else:
                unlabled_data = torch.cat([unlabled_data, input])

    labeled_dataset = TensorDataset(labeled_data, labeled_labels)
    labeled_trainloader = DataLoader(labeled_dataset, batch_size=64, shuffle=True)
    unlabeled_dataset = TensorDataset(unlabled_data)
    unlabeled_trainloader = DataLoader(unlabeled_dataset, batch_size=64, shuffle=True)
    model.train()
    return labeled_trainloader, unlabeled_trainloader

def calculate_accuracy(list1, list2):
    count = 0
    total = len(list1)
    for i in range(total):
        if list1[i] == list2[i]:
            count += 1
    accuracy = count / total
    print(accuracy)


def fit(dataset, model):
    labeled_indexes, unlabeled_indexes = initial_split(train_loader=dataset['train_loader'])
    orig_labels = dataset['train_loader'].dataset.targets

    # create labeled trainloader
    images = []
    labels = []
    for i in labeled_indexes:
        img, label = dataset['train_loader'].dataset[i]
        images.append(img)
        labels.append(label)

    images_tensor = torch.stack(images)
    labels_tensor = torch.tensor(labels)

    labeled_dataset = TensorDataset(images_tensor, labels_tensor)
    labeled_trainloader = DataLoader(labeled_dataset, batch_size=64, shuffle=True)

    # create unlabeled trainloader
    images = []
    for i in unlabeled_indexes:
        img, label = dataset['train_loader'].dataset[i]
        images.append(img)

    images_tensor = torch.stack(images)

    unlabeled_dataset = torch.utils.data.TensorDataset(images_tensor)
    unlabeled_trainloader = torch.utils.data.DataLoader(unlabeled_dataset, batch_size=64, shuffle=True)

    # train ssl
    while True:
        unlabeled_samples = unlabeled_trainloader.dataset.tensors[0].shape[0]
        print(f'unlabeled samples: {len(unlabeled_samples)}')
        dataset_ssl = {
            'train_loader': labeled_trainloader,
            'test_loader': dataset['test_loader']
        }
        train.moe_ssl_train(model, dataset_ssl)
        labeled_trainloader, unlabeled_trainloader = label_samples(model, unlabeled_trainloader, labeled_trainloader, th=setup['ssl_th'])
        unlabeled_samples = unlabeled_trainloader.dataset.tensors[0].shape[0]
        if len(unlabeled_samples) == 0:
            break

    print(f'unlabeled samples: {len(unlabeled_samples)}')
    print(f'ssl labels: {labeled_trainloader.dataset.targets[:100]}')
    print(f'GT labels: {orig_labels[:100]}')
    # calculate_accuracy(labeled_trainloader.dataset.targets, orig_labels)
    return model, labeled_trainloader
