import numpy as np
from torch.utils.data import DataLoader, random_split
import torch
import torch.nn.functional as F
import train
from torch.utils.data import Subset
from config import setup
import torchvision.transforms as transforms
from utils import CustomDataset
import numpy as np


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

def split_labeled_unlabeled_data(n_labeled_samples, data_loader):

    # Step 1: Get all samples and corresponding labels
    all_samples = data_loader.dataset.data
    all_labels = data_loader.dataset.targets

    # Step 3: Create DataLoader with standard CIFAR10 transforms for labeled data
    labeled_samples = all_samples[:n_labeled_samples]
    labeled_labels = all_labels[:n_labeled_samples]
    labeled_dataset = CustomDataset(labeled_samples, labeled_labels, transform=transform_train)
    labeled_data_loader = torch.utils.data.DataLoader(labeled_dataset, batch_size=data_loader.batch_size)

    # Step 4: Create DataLoader with standard CIFAR10 transforms for unlabeled data
    unlabeled_samples = all_samples[n_labeled_samples:]
    unlabeled_labels = all_labels[n_labeled_samples:]
    unlabeled_dataset = CustomDataset(unlabeled_samples, unlabeled_labels, transform=transform_train)
    unlabeled_data_loader = torch.utils.data.DataLoader(unlabeled_dataset, batch_size=data_loader.batch_size)

    return labeled_data_loader, unlabeled_data_loader

def get_high_score_unlabeled_data(unlabeled_data_loader, model, threshold):
    unlabeled_samples = torch.tensor([])
    ssl_labeled_samples = torch.tensor([])
    ssl_labeled_labels = torch.tensor([])

    # Set the model to evaluation mode
    model.eval()
    model = model.to(device)

    with torch.no_grad():
        i=1
        for batch_samples, _ in unlabeled_data_loader:
            i+=1
            batch_samples = batch_samples.to(device)
            predictions = model(batch_samples)[0]
            predictions = F.softmax(predictions, dim=1)

            # Calculate the maximum prediction score and predicted labels for each sample in the batch
            max_scores, predicted_labels = torch.max(predictions, dim=1)

            # Filter samples that have a prediction score higher than the threshold
            high_score_indices = (max_scores > threshold).nonzero().squeeze()
            low_score_indices = (max_scores <= threshold).nonzero().squeeze()

            # Append the high-score samples and their labels to the filtered lists
            if high_score_indices.numel() > 0:
                high_score_samples = batch_samples[high_score_indices]
                high_score_labels = predicted_labels[high_score_indices]
                if high_score_indices.shape[0] == 1:
                    high_score_samples = high_score_samples.unsqueeze(0)
                ssl_labeled_samples = torch.cat((ssl_labeled_samples, high_score_samples), dim=0)
                ssl_labeled_labels = torch.cat((ssl_labeled_labels, high_score_labels), dim=0)

            # Append the remaining low-score samples and their labels back to the unlabeled lists
            if low_score_indices.numel() > 0:
                low_score_samples = batch_samples[low_score_indices]
                if low_score_indices.shape[0] == 1:
                    low_score_samples = low_score_samples.unsqueez(0)
                unlabeled_samples = torch.cat((unlabeled_samples, low_score_samples), dim=0)
            if i == 10:
                break

    # Update the unlabeled_data_loader with the remaining low-score samples
    if unlabeled_samples.shape[0] > 0:
        remaining_unlabeled_samples = unlabeled_samples.numpy()
        remaining_unlabeled_samples = np.transpose(remaining_unlabeled_samples, (0, 2, 3, 1))
        remaining_unlabeled_labels = np.zeros(remaining_unlabeled_samples.shape[0]).tolist()


        remaining_unlabeled_dataset = CustomDataset(remaining_unlabeled_samples, remaining_unlabeled_labels, transform=transform_train)
        unlabeled_data_loader = DataLoader(remaining_unlabeled_dataset, batch_size=unlabeled_data_loader.batch_size)

    return ssl_labeled_samples, ssl_labeled_labels, unlabeled_data_loader

def add_unlabeled_samples_labels_to_labeled_data_loader(ssl_labeled_samples, ssl_labeled_labels, labeled_data_loader):
    samples_loader = labeled_data_loader.dataset.samples
    labels_loader = labeled_data_loader.dataset.labels
    ssl_labeled_samples = ssl_labeled_samples.numpy()
    ssl_labeled_samples = np.transpose(ssl_labeled_samples, (0, 2, 3, 1))

    combined_samples = np.concatenate((samples_loader, ssl_labeled_samples), axis=0)
    combined_labels = np.concatenate((labels_loader, ssl_labeled_labels), axis=0).tolist()
    combined_labels = [int(x) for x in combined_labels]

    labeled_dataset = CustomDataset(combined_samples, combined_labels, transform=transform_train)
    labeled_dataloader = DataLoader(labeled_dataset, batch_size= 128)

    return labeled_dataloader




def propagate_labels(labeled_data_loader, unlabeled_data_loader, model, th):
    ssl_labeled_samples, ssl_labeled_labels, unlabeled_data_loader = get_high_score_unlabeled_data(unlabeled_data_loader, model, th)
    labeled_data_loader = add_unlabeled_samples_labels_to_labeled_data_loader(ssl_labeled_samples, ssl_labeled_labels, labeled_data_loader)
    return labeled_data_loader, unlabeled_data_loader





def fit(dataset, model):
    while len(dataset['labeled_train_loader'].dataset) < 49_000:
        model.train()
        train.moe_train_ssl(model, dataset)
        labeled_data_loader, unlabeled_data_loader = propagate_labels(dataset['labeled_train_loader'], dataset['unlabeled_train_loader'], model, setup['ssl_th'])
        dataset['labeled_train_loader'] = labeled_data_loader
        dataset['unlabeled_train_loader'] = unlabeled_data_loader
        print(f'labeled dataloader length: {len(labeled_data_loader)}, ublabeled dataloader length: {len(unlabeled_data_loader)}')
    return dataset