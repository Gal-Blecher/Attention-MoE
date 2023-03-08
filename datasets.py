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
import random

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
        print('==> Preparing data..')

        # def seed_worker(worker_id):
        #     worker_seed = torch.initial_seed() % 2 ** 32
        #     np.random.seed(worker_seed)
        #     random.seed(worker_seed)
        #
        # g = torch.Generator()
        # g.manual_seed(0)

        transform_train = transforms.Compose([
            transforms.CenterCrop(24),
            transforms.Resize(size=32),
            transforms.ToTensor(),
            transforms.GaussianBlur(kernel_size=(3, 7), sigma=(1.1, 2.2)),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.CenterCrop(24),
            transforms.Resize(size=32),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_rotate_train = transforms.Compose([
            torchvision.transforms.RandomRotation((30, 30)),
            transforms.CenterCrop(24),
            transforms.Resize(size=32),
            transforms.ToTensor(),
            transforms.GaussianBlur(kernel_size=(3, 7), sigma=(1.1, 2.2)),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_rotate_test = transforms.Compose([
            torchvision.transforms.RandomRotation((30, 30)),
            transforms.CenterCrop(24),
            transforms.Resize(size=32),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        # Create trainset
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=True, transform=transform_train)

        # Create cluster and targets
        trainset.targets = torch.tensor(trainset.targets)
        trainset.cluster = trainset.targets
        trainset.targets = torch.zeros_like(trainset.targets)

        # trainset negative examples
        trainset_flip = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                     download=True, transform=transform_rotate_train)
        # Cluster and targets
        trainset_flip.targets = torch.tensor(trainset_flip.targets)
        trainset_flip.cluster = trainset_flip.targets
        trainset_flip.targets = torch.ones_like(trainset_flip.targets)

        trainset = torch.utils.data.ConcatDataset([trainset, trainset_flip])
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                                  shuffle=True, num_workers=2)

        # Testset cluster and targets
        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                               download=True, transform=transform_test)
        testset.targets = torch.tensor(testset.targets)
        testset.cluster = testset.targets
        testset.targets = torch.zeros_like(testset.targets)

        # Testset negative
        testset_flip = torchvision.datasets.CIFAR10(root='./data', train=False,
                                                    download=True, transform=transform_rotate_test)
        testset_flip.targets = torch.tensor(testset_flip.targets)
        testset_flip.cluster = testset_flip.targets
        testset_flip.targets = torch.ones_like(testset_flip.targets)

        testset = torch.utils.data.ConcatDataset([testset, testset_flip])
        test_loader = torch.utils.data.DataLoader(testset, batch_size=100,
                                                 shuffle=True, num_workers=2)

        classes = ('plane', 'car', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck')


    # print_data_info(train_loader, test_loader, setup['dataset_name'])

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


