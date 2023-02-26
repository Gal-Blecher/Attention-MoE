from torchvision import transforms
import torchvision
import torch
from torchvision.datasets import MNIST
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as transforms

import config
import utils
from config import setup

def get_dataset(dataset_name=None):
    if dataset_name != None:
        setup['dataset_name'] = dataset_name
    if setup['dataset_name'] == 'cifar10':
        print(f'==> Preparing data cifar10')
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        trainset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform_train)
        train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=128, shuffle=True, num_workers=2)

        testset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(
            testset, batch_size=100, shuffle=False, num_workers=2)

        classes = ('plane', 'car', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck')

    if setup['dataset_name'] == 'cifar100':
        print(f'==> Preparing data cifar100')
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        trainset = torchvision.datasets.CIFAR100(
            root='./data', train=True, download=True, transform=transform_train)
        train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=128, shuffle=True, num_workers=2)

        testset = torchvision.datasets.CIFAR100(
            root='./data', train=False, download=True, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(
            testset, batch_size=100, shuffle=False, num_workers=2)

    if setup['dataset_name'] == 'mnist':
        print(f'==> Preparing data MNIST')
        train_dataset = MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
        test_dataset = MNIST(root='./data', train=False, transform=transforms.ToTensor())
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    if setup['dataset_name'] == 'cub200':
        print(f'==> Preparing data CUB200')
        transform_train = transforms.Compose([
                transforms.RandomResizedCrop(448),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                     std=(0.229, 0.224, 0.225))
            ])
        transform_test = transforms.Compose([
                transforms.Resize(int(448/0.875)),
                transforms.CenterCrop(448),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                     std=(0.229, 0.224, 0.225))
            ])



        train_data = utils.Cub200('./cub2011', train=True, transform=transform_train)
        test_data = utils.Cub200('./cub2011', train=False, transform=transform_test)

        train_loader = torch.utils.data.DataLoader(train_data, batch_size=config.train_config['dataset']['cub200']['batch_size'], shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=config.train_config['dataset']['cub200']['batch_size'], shuffle=False)

    if setup['dataset_name'] == 'rotate_cifar10':
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])


        trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

        # Create a copy of the dataset with rotated images
        rotated_trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        rotated_testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

        # Rotate each image in the original training set by 30 degrees
        for i, (image, label) in enumerate(trainset):
            plot_image(image)
            rotated_image = transforms.functional.rotate(image, 30)
            rotated_image = rotated_image.permute(1, 2, 0)  # Transpose to (32, 32, 3)
            rotated_trainset.data[i] = rotated_image
            rotated_trainset.targets[i] = 1
        for i, (image, label) in enumerate(testset):
            rotated_image = transforms.functional.rotate(image, 30)
            rotated_image = rotated_image.permute(1, 2, 0)  # Transpose to (32, 32, 3)
            rotated_testset.data[i] = rotated_image
            rotated_testset.targets[i] = 1

        # Set the labels for the original images to 0
        trainset.targets = torch.zeros(len(trainset))

        # Combine the original and rotated datasets
        combined_trainset = torch.utils.data.ConcatDataset([trainset, rotated_trainset])
        combined_testset = torch.utils.data.ConcatDataset([testset, rotated_testset])

        # Create data loaders for the combined dataset
        train_loader = torch.utils.data.DataLoader(combined_trainset, batch_size=128, shuffle=True, num_workers=2)
        test_loader = torch.utils.data.DataLoader(combined_testset, batch_size=64, shuffle=False, num_workers=2)
        show_rotated_images(train_loader)

    print_data_info(train_loader, test_loader, setup['dataset_name'])

    dataset = {'train_loader': train_loader,
               'test_loader': test_loader}

    return dataset

def print_data_info(train_loader, test_loader, dataset_name):
    if dataset_name == 'cub200':
        train_class_counts = torch.unique(torch.tensor(train_loader.dataset.train_label), return_counts=True)[1]
        test_class_counts = torch.unique(torch.tensor(test_loader.dataset.test_label), return_counts=True)[1]
    else:
        train_class_counts = torch.unique(torch.tensor(train_loader.dataset.targets), return_counts=True)[1]
        test_class_counts = torch.unique(torch.tensor(test_loader.dataset.targets), return_counts=True)[1]
    print(f'train loader length: {train_loader.dataset.__len__()}')
    print(f'Instances per class in the train loader: \n {train_class_counts}')
    print(f'train loader length: {test_loader.dataset.__len__()}')
    print(f'Instances per class in the test loader: \n {test_class_counts}')

def show_rotated_images(trainloader):
    # Get a batch of images from the trainloader
    data_iter = iter(trainloader)
    images, labels = data_iter.next()

    # Find the indices of images that have been rotated
    rotated_indices = np.argwhere(labels == 1).flatten()

    # Sample 5 pairs of original and rotated images
    sample_indices = np.random.choice(rotated_indices, size=5, replace=False)

    # Plot the original and rotated images side by side
    fig, axs = plt.subplots(nrows=5, ncols=2, figsize=(10, 15))

    for i, index in enumerate(sample_indices):
        original_index = index // 2  # Get the index of the original image
        axs[i][0].imshow(np.transpose(images[original_index], (1, 2, 0)))
        axs[i][0].set_title('Original Image')
        axs[i][0].axis('off')
        axs[i][1].imshow(np.transpose(images[index], (1, 2, 0)))
        axs[i][1].set_title('Rotated Image')
        axs[i][1].axis('off')

    plt.tight_layout()
    plt.show()


def plot_image(image_tensor):
    image_tensor = np.transpose(image_tensor, (1, 2, 0))
    # Check if the input tensor has 3 channels
    if image_tensor.shape[-1] != 3:
        print("Error: Input tensor must have 3 channels")
        return

    # Rescale the pixel values to be between 0 and 1
    # image_tensor = (image_tensor - np.min(image_tensor)) / (np.max(image_tensor) - np.min(image_tensor))

    # Create a new figure and plot the image
    plt.figure(figsize=(8, 8))
    plt.imshow(image_tensor)
    plt.axis('off')
    plt.show()


